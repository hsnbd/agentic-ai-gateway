module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  parser: '@typescript-eslint/parser',
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module', ecmaFeatures: { jsx: true } },
  plugins: ['@typescript-eslint', 'react-hooks'],
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended', 'plugin:react-hooks/recommended'],
  ignorePatterns: ['../app/ui_static', 'node_modules'],
  rules: {
    // A render-time read of a later `const` (e.g. a useState value used in a
    // sort) is a TDZ crash that TypeScript cannot see. It once took down the
    // whole RAG page.
    '@typescript-eslint/no-use-before-define': [
      'error',
      { functions: false, classes: false, variables: true, typedefs: false, ignoreTypeReferences: true },
    ],
  },
};
