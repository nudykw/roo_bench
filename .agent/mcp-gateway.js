const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const net = require('net');
const { spawn, execSync } = require('child_process');

const projectRoot = path.resolve(__dirname, '..');
const hash = crypto.createHash('md5').update(projectRoot).digest('hex');
const port = 12000 + (parseInt(hash.substring(0, 8), 16) % 1000);

if (process.argv.includes('--daemon')) {
  runDaemon(port);
} else {
  runClient(port);
}

// ==================== CLIENT MODE ====================
function runClient(port) {
  const client = net.connect({ port }, () => {
    process.stdin.pipe(client);
    client.pipe(process.stdout);
  });

  client.on('error', () => {
    // Daemon is not running, let's start it in the background
    const logPath = path.join(__dirname, 'daemon.log');
    const out = fs.openSync(logPath, 'a');
    const err = fs.openSync(logPath, 'a');

    const daemon = spawn(process.execPath, [__filename, '--daemon'], {
      detached: true,
      stdio: ['ignore', out, err],
      cwd: projectRoot
    });
    daemon.unref();

    // Start polling to connect
    connectToDaemon(port);
  });

  client.on('end', () => {
    process.exit(0);
  });
}

function connectToDaemon(port, retries = 60) {
  const client = net.connect({ port }, () => {
    process.stdin.pipe(client);
    client.pipe(process.stdout);
  });

  client.on('error', (err) => {
    if (retries > 0) {
      setTimeout(() => connectToDaemon(port, retries - 1), 100);
    } else {
      console.error('Failed to connect to MCP daemon:', err);
      process.exit(1);
    }
  });

  client.on('end', () => {
    process.exit(0);
  });
}

// ==================== DAEMON MODE ====================
function runDaemon(port) {
  console.log(`Starting codegraph MCP Daemon on port ${port}...`);
  console.log(`Project root: ${projectRoot}`);

  const activeSockets = new Set();
  const idMap = new Map(); // globalId -> { socket, originalId }
  const requestQueue = [];
  
  let globalIdCounter = 0;
  let codegraphProcess = null;
  let watcherProcess = null;
  let isSyncing = false;
  let idleTimer = null;

  // Initialize codegraph if not done yet
  const codegraphDir = path.join(projectRoot, '.codegraph');
  if (!fs.existsSync(codegraphDir)) {
    console.log('Initializing codegraph for the first time...');
    try {
      execSync('codegraph init', { cwd: projectRoot, stdio: 'inherit' });
      console.log('Running initial full index...');
      execSync('codegraph index', { cwd: projectRoot, stdio: 'inherit' });
    } catch (e) {
      console.error('Error during codegraph initialization:', e);
    }
  }

  // Start the background server and inotify watcher
  startCodegraphServe();
  startInotifyWatcher();
  startIdleTimer();

  const server = net.createServer((socket) => {
    resetIdleTimer();
    activeSockets.add(socket);
    console.log(`Client connected. Total active clients: ${activeSockets.size}`);

    const handleLine = createLineBuffer((line) => {
      try {
        const msg = JSON.parse(line);
        if (msg && typeof msg === 'object') {
          if ('id' in msg) {
            const globalId = ++globalIdCounter;
            idMap.set(globalId, { socket, originalId: msg.id });
            msg.id = globalId;
            sendToCodegraph(JSON.stringify(msg));
          } else {
            // Notification, forward directly
            sendToCodegraph(line);
          }
        }
      } catch (e) {
        console.error('Error processing client message:', e, line);
      }
    });

    socket.on('data', handleLine);

    socket.on('error', (err) => {
      console.error('Client socket error:', err.message);
    });

    socket.on('close', () => {
      activeSockets.delete(socket);
      console.log(`Client disconnected. Total active clients: ${activeSockets.size}`);
      
      // Clean up outstanding mappings for this socket to avoid memory leak
      for (const [globalId, mapping] of idMap.entries()) {
        if (mapping.socket === socket) {
          idMap.delete(globalId);
        }
      }

      if (activeSockets.size === 0) {
        startIdleTimer();
      }
    });
  });

  server.listen(port, '127.0.0.1', () => {
    console.log(`Daemon successfully listening on 127.0.0.1:${port}`);
  });

  // Helpers
  function createLineBuffer(onLine) {
    let buffer = '';
    return (chunk) => {
      buffer += chunk.toString();
      let lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.trim()) {
          onLine(line);
        }
      }
    };
  }

  function startCodegraphServe() {
    console.log('Spawning codegraph serve process...');
    codegraphProcess = spawn('codegraph', ['serve', '--mcp', '--path', projectRoot], {
      cwd: projectRoot,
      stdio: ['pipe', 'pipe', 'inherit']
    });

    const handleLine = createLineBuffer((line) => {
      try {
        const msg = JSON.parse(line);
        if (msg && typeof msg === 'object' && 'id' in msg) {
          const mapping = idMap.get(msg.id);
          if (mapping) {
            msg.id = mapping.originalId;
            mapping.socket.write(JSON.stringify(msg) + '\n');
            idMap.delete(msg.id);
          }
        } else {
          // Broadcast server notifications to all active clients
          for (const socket of activeSockets) {
            socket.write(line + '\n');
          }
        }
      } catch (e) {
        console.error('Error processing codegraph response:', e, line);
      }
    });

    codegraphProcess.stdout.on('data', handleLine);

    codegraphProcess.on('exit', (code) => {
      console.log(`codegraph serve exited with code ${code}`);
      codegraphProcess = null;
      if (!isSyncing) {
        // Auto restart if it crashed and we are not syncing
        setTimeout(startCodegraphServe, 1000);
      }
    });

    // Flush any requests queued while starting/syncing
    flushRequestQueue();
  }

  function sendToCodegraph(messageStr) {
    if (codegraphProcess && !isSyncing) {
      codegraphProcess.stdin.write(messageStr + '\n');
    } else {
      requestQueue.push(messageStr);
    }
  }

  function flushRequestQueue() {
    while (requestQueue.length > 0 && codegraphProcess && !isSyncing) {
      const msg = requestQueue.shift();
      codegraphProcess.stdin.write(msg + '\n');
    }
  }

  function startInotifyWatcher() {
    console.log('Spawning inotifywait file watcher...');
    watcherProcess = spawn('inotifywait', [
      '-m', '-r',
      '-e', 'modify,create,delete,move',
      '--exclude', '(bin|obj|\\.git|\\.codegraph)',
      projectRoot
    ]);

    let syncTimeout = null;
    watcherProcess.stdout.on('data', (data) => {
      if (syncTimeout) clearTimeout(syncTimeout);
      syncTimeout = setTimeout(triggerSync, 3000); // 3 seconds debounce
    });

    watcherProcess.on('exit', (code) => {
      console.log(`inotifywait exited with code ${code}`);
      watcherProcess = null;
    });
  }

  function triggerSync() {
    if (isSyncing) return;
    isSyncing = true;
    console.log('File changes detected. Pausing codegraph serve for incremental sync...');
    
    if (codegraphProcess) {
      codegraphProcess.removeAllListeners('exit');
      codegraphProcess.on('exit', () => {
        codegraphProcess = null;
        executeSync();
      });
      codegraphProcess.kill();
    } else {
      executeSync();
    }
  }

  function executeSync() {
    console.log('Running codegraph sync...');
    const syncProc = spawn('codegraph', ['sync'], { cwd: projectRoot });
    
    syncProc.on('exit', (code) => {
      console.log(`codegraph sync completed with code ${code}`);
      isSyncing = false;
      startCodegraphServe();
    });
  }

  function resetIdleTimer() {
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  }

  function startIdleTimer() {
    resetIdleTimer();
    idleTimer = setTimeout(() => {
      console.log('No active connections for 15 minutes. Shutting down daemon...');
      if (codegraphProcess) codegraphProcess.kill();
      if (watcherProcess) watcherProcess.kill();
      server.close(() => {
        process.exit(0);
      });
    }, 15 * 60 * 1000); // 15 minutes
  }
}
