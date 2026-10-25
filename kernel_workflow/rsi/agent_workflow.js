export const meta = { name: 'rsi-knowledge-agent', description: 'One isolated knowledge developer or reviewer call.' };

if (!args.prompt || !args.schema || !args.model) throw new Error('RSI agent input is incomplete');
return await agent(args.prompt, {
  label: args.role,
  phase: 'KnowledgeEvolution',
  model: args.model,
  cwd: args.cwd,
  schema: args.schema,
});
