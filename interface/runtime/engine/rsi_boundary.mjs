import { createHash, randomUUID } from 'node:crypto';
import { lstat, mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.keys(value).sort().map(key => [key, canonical(value[key])])
  );
  return value;
}

const bytes = value => Buffer.from(JSON.stringify(canonical(value)));
const sha = payload => createHash('sha256').update(payload).digest('hex');

async function syncDirectory(path) {
  const handle = await open(path, 'r');
  try { await handle.sync(); } finally { await handle.close(); }
}

async function atomic(path, payload) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.pending`;
  const handle = await open(temporary, 'wx', 0o600);
  try {
    await handle.writeFile(payload);
    await handle.sync();
  } finally { await handle.close(); }
  try { await rename(temporary, path); }
  finally {
    try { await unlink(temporary); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  await syncDirectory(dirname(path));
}

function inside(path, root) {
  const rel = relative(root, path);
  return rel === '' || (rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

export function createRSIRecorder(config) {
  if (!config || !isAbsolute(config.directory) || !Array.isArray(config.allowed_roots)
      || !Array.isArray(config.source_paths)) throw new Error('invalid RSI capture configuration');
  const directory = resolve(config.directory);
  const roots = config.allowed_roots.map(root => resolve(root));
  const sourcePaths = config.source_paths;
  for (const path of sourcePaths) {
    if (typeof path !== 'string' || isAbsolute(path) || path.split(/[\\/]/).some(part => !part || part === '..' || part === '.')) {
      throw new Error('RSI source paths must be safe relative paths');
    }
  }
  let boundarySequence = 0;
  let agentSequence = 0;

  async function safePath(path) {
    const resolved = resolve(path);
    if (!roots.some(root => inside(resolved, root))) throw new Error('RSI capture path is outside declared roots');
    let current = resolved;
    while (true) {
      try {
        if ((await lstat(current)).isSymbolicLink()) throw new Error('RSI source path is a symlink');
      } catch (error) { if (error.code !== 'ENOENT') throw error; }
      if (dirname(current) === current) break;
      current = dirname(current);
    }
    return resolved;
  }

  async function object(payload) {
    const hash = sha(payload);
    const path = resolve(directory, 'objects', hash);
    try {
      const existing = await readFile(path);
      if (!existing.equals(payload)) throw new Error('RSI immutable object changed');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      await atomic(path, payload);
    }
    return { sha256: hash, size: payload.length, path };
  }

  async function snapshot(workspace) {
    const root = await safePath(workspace);
    const files = {};
    const missing = [];
    async function visit(path) {
      const target = await safePath(path);
      let stat;
      try { stat = await lstat(target); }
      catch (error) {
        if (error.code !== 'ENOENT') throw error;
        missing.push(relative(root, target));
        return;
      }
      if (stat.isDirectory()) {
        for (const entry of (await readdir(target)).sort()) {
          if (!['.git', '__pycache__'].includes(entry)) await visit(resolve(target, entry));
        }
      } else if (stat.isFile()) {
        const payload = await readFile(target);
        const after = await lstat(target);
        if (stat.mtimeMs !== after.mtimeMs || stat.size !== after.size) throw new Error('RSI source changed during capture');
        const stored = await object(payload);
        files[relative(root, target).split(sep).join('/')] = { sha256: stored.sha256, size: stored.size };
      } else throw new Error('RSI source is not a regular file or directory');
    }
    for (const path of sourcePaths) await visit(resolve(root, path));
    return { workspace: root, files, missing, fingerprint: missing.length || !Object.keys(files).length ? null : sha(bytes(files)) };
  }

  return {
    async beginAgent(prompt, options, attempt) {
      const id = `agent-${++agentSequence}`;
      await atomic(resolve(directory, `${id}-input.json`), bytes({ id, prompt, label: options.label,
        phase: options.phase, model: options.model || null, attempt, schema: options.schema || null }));
      return id;
    },
    async endAgent(id, value) {
      await atomic(resolve(directory, `${id}-output.json`), bytes({ id, ...value }));
    },
    async boundary(value) {
      if (!['plan_dispatch', 'candidate_verify', 'commit_director'].includes(value.kind)) {
        throw new Error('unsupported RSI boundary');
      }
      const sequence = ++boundarySequence;
      const sources = {};
      for (const [name, workspace] of Object.entries(value.workspaces || {})) sources[name] = await snapshot(workspace);
      const artifacts = [];
      for (const path of value.artifacts || []) {
        const resolved = await safePath(path);
        try { artifacts.push({ source: resolved, ...await object(await readFile(resolved)) }); }
        catch (error) {
          if (error.code !== 'ENOENT') throw error;
          artifacts.push({ source: resolved, missing: true });
        }
      }
      const document = { schema_version: 1, sequence, ...value, sources, artifacts };
      const path = resolve(directory, `boundary-${sequence}-${value.kind}.json`);
      const payload = bytes(document);
      await atomic(path, payload);
      return { path, sha256: sha(payload), kind: 'observed', start_line: 1, end_line: 1 };
    },
  };
}
