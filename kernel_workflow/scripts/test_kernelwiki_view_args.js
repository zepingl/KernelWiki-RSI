#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The KernelWiki-view args of kernel_lane.js (learned_kb_dir, learned_inbox_dir, run_id) are opt-in:
// without them no prompt mentions them; with them the readers of learned cards are pointed at the view,
// and the deep engineer's read line names it. Runs the REAL lane with stubbed runtime globals.
//
// Run:  node kernel_workflow/scripts/test_kernelwiki_view_args.js [path/to/kernel_lane.js]
'use strict';

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const lanePath = process.argv[2] || path.join(wfDir, 'kernel_lane.js');
const body = fs.readFileSync(lanePath, 'utf8').replace(/^export const meta/m, 'const meta');

async function run(extraArgs) {
  const prompts = [];
  const globals = {
    args: { kernel_path: '/tmp/kernel', workflow_dir: wfDir, budget: 2, expected_gfx: 'gfx950',
            use_learned_kb: 'true', warm_start: 'off', ...extraArgs },
    phase: () => {},
    log: () => {},
    workflow: async () => null,
    parallel: async (items) => Promise.all(items.map((item) => item())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (item, index) => {
      let value = item;
      for (const stage of stages) value = await stage(value, item, index);
      return value;
    })),
    budget: { total: null, spent: () => 0, remaining: () => 0 },
    agent: async (prompt, options) => {
      prompts.push({ label: options.label, prompt });
      if (options.label === 'director:setup') {
        return { eval_dir: '/tmp/eval', workspace: '/tmp/eval/workspace', baseline_dir: '/tmp/eval/baseline',
                 kernel_name: 'kernel', device_name: 'fixture', baseline_frozen: true, source_files: ['kernel.py'],
                 device_gfx: 'gfx950', device_target: 'unknown', physical_cu_count: 256 };
      }
      if (options.label === 'tech_lead:analyze') {
        return { kernel_type: 'triton', modifiable_files: ['kernel.py'], kk_operator: 'dense_gemm',
                 kk_language: 'triton', kk_refs: [] };
      }
      if (options.label === 'benchmark_engineer') {
        return { commandment_path: '/tmp/eval/COMMANDMENT.md', baseline_per_case: [{ name: 'x', latency_ms: 1 }],
                 baseline_geomean_ms: 1 };
      }
      if (options.label === 'profile_engineer:baseline') return { bottleneck: 'unknown', summary_path: '' };
      if (options.label.startsWith('tech_lead:plan')) {
        return { stop: false, directions: [{ id: 'deep', title: 'deep', specialty: 'deep_explore', prompt: 'x' }] };
      }
      return null;
    },
  };
  const fn = new Function(...Object.keys(globals), `return (async () => { ${body} })();`);
  try { await fn(...Object.values(globals)); } catch (e) { /* a stubbed run may end early; prompts are what we test */ }
  return prompts;
}

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };
const find = (prompts, prefix) => prompts.filter((entry) => entry.label.startsWith(prefix)).map((e) => e.prompt).join('\n');

(async () => {
  if (process.argv[2]) {   // dump mode: print every prompt, for byte-for-byte comparison across revisions
    for (const entry of await run({})) console.log(JSON.stringify(entry));
    return;
  }
  const plain = await run({});
  ok(!plain.some((e) => /LEARNED_KB_DIR|LEARNED_KB_SOURCE|RUN_ID|KernelWiki view/.test(e.prompt)),
    'default run: no prompt mentions the KernelWiki view, its inbox or a run id');

  const view = '/tmp/kernelwiki-view/learned';
  const viewed = await run({ learned_kb_dir: view + '/', learned_inbox_dir: '/tmp/inbox', run_id: 'run-7' });
  const plan = find(viewed, 'tech_lead:plan');
  ok(plan.includes(`LEARNED_KB_DIR: ${view}`) && plan.includes(`read ${view}/ instead`),
    'planner is pointed at the view when learned_kb_dir is set');
  ok(viewed.some((e) => e.prompt.includes(`replaced by the KernelWiki view ${view}/`)),
    "deep engineer's read line names the view");

  const off = await run({ learned_kb_dir: view, use_learned_kb: 'false' });
  ok(!off.some((e) => e.prompt.includes('LEARNED_KB_DIR') || e.prompt.includes('KernelWiki view')),
    'learned KB off: the view is never offered');

  if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
  console.log('all green');
})();
