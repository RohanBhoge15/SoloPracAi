declare module "*.cjs" {
  const factory: (opts?: Record<string, unknown>) => Promise<unknown>;
  export default factory;
}
