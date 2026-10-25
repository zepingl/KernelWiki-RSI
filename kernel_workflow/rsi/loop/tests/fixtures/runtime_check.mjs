import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const repo = process.argv[2];
const { createRuntime } = await import(pathToFileURL(join(repo, 'interface/runtime/engine/run_workflow.mjs')));
const temporary = await mkdtemp(join(tmpdir(), 'geak-rsi-runtime-'));
try {
  const parent = join(temporary, 'parent');
  const output = join(temporary, 'output');
  await mkdir(join(parent, 'kernel_src'), { recursive: true });
  await writeFile(join(parent, 'kernel_src/k.py'), 'value = 1\n');
  const labels = [];
  const backend = { name: 'fake', async runAgent({ label, prompt, model }) {
    labels.push(label);
    assert.equal(model, 'claude-opus-5.5');
    if (label === 'rsi:plan') return { text: JSON.stringify({ stop: false, directions: [{ id: 'one',
      title: 'Fixture direction', specialty: 'compute', prompt: 'Fixture', learned_refs: ['original-card.md'] }] }) };
    assert.ok(label.startsWith('eng '));
    assert.ok(prompt.includes('INSIGHTS: []'));
    await mkdir(join(output, 'direction/workspace/kernel_src'), { recursive: true });
    await writeFile(join(output, 'direction/workspace/kernel_src/k.py'), 'value = 2\n');
    await writeFile(join(output, 'direction/best_patch.diff'), 'fixture patch\n');
    return { text: JSON.stringify({ status: 'improved', speedup_geomean: 9, measurement_valid: true }) };
  } };
  const args = { kernel_path: parent, workflow_dir: join(repo, 'kernel_workflow'),
    rsi_generation: 'normal', gpu_ids: '0', use_learned_kb: 'false',
    rsi_context: { CANONICAL: parent, EVAL_DIR: output, COMMANDMENT: join(temporary, 'COMMANDMENT.md'),
      BASELINE_PER_CASE: [], profileSummary: null, history: { insights: [] },
      KERNEL_KNOWLEDGE_DIR: '', KK_OPERATOR: '', KK_LANGUAGE: '', KK_REFS: [],
      GFX: 'gfx950', DEVICE_TARGET: '', PHYSICAL_CU_COUNT: 0, ROOFLINE_STATUS: 'unknown' },
    rsi_capture: { directory: join(temporary, 'capture'), allowed_roots: [temporary], source_paths: ['kernel_src'] } };
  const runtime = createRuntime({ backend, schemaRetries: 0, log: () => {} });
  const result = await runtime.runScript(join(repo, 'kernel_workflow/kernel_lane.js'), args, 0);
  assert.equal(result.status, 'generated');
  assert.equal(result.verification, 'host_evaluator_required');
  assert.deepEqual(labels, ['rsi:plan', 'eng one:compute']);
  const dispatch = JSON.parse(await readFile(result.dispatch.path, 'utf8'));
  assert.deepEqual(dispatch.plan.directions[0].learned_refs, ['original-card.md']);
  const candidate = JSON.parse(await readFile(result.result.evidence.path, 'utf8'));
  assert.notEqual(candidate.sources.parent.fingerprint, candidate.sources.candidate.fingerprint);
  assert.equal(candidate.invocation, 'agent-2');
  const input = JSON.parse(await readFile(join(temporary, 'capture/agent-2-input.json'), 'utf8'));
  assert.ok(input.prompt.includes('OUTPUT CONTRACT'));
  assert.equal(input.model, 'claude-opus-5.5');
  await assert.rejects(runtime.runScript(join(repo, 'kernel_workflow/kernel_lane.js'),
    { ...args, rsi_capture: undefined }, 0), /durable host capture/);
  await symlink(join(parent, 'kernel_src/k.py'), join(parent, 'kernel_src/link.py'));
  const second = createRuntime({ backend, schemaRetries: 0, log: () => {} });
  await assert.rejects(second.runScript(join(repo, 'kernel_workflow/kernel_lane.js'),
    { ...args, rsi_capture: { ...args.rsi_capture, directory: join(temporary, 'capture-symlink') } }, 0), /symlink/);
  process.stdout.write('RSI runtime boundaries and restricted lane passed\n');
} finally { await rm(temporary, { recursive: true, force: true }); }
