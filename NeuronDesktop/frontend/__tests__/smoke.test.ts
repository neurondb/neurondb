/**
 * Smoke test so npm test runs actual tests (not only lint).
 */
describe('smoke', () => {
  it('runs', () => {
    expect(1 + 1).toBe(2);
  });
});
