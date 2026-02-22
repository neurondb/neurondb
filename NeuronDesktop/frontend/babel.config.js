/** Babel config for Jest so .tsx test files parse (JSX + TypeScript). */
module.exports = {
  presets: [
    ['@babel/preset-react', { runtime: 'automatic' }],
    ['@babel/preset-typescript', { allowDeclareFields: true }],
  ],
};
