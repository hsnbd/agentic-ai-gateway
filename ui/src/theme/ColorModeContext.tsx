import { createContext, useContext, useMemo, useState, type PropsWithChildren } from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import type { PaletteMode } from '@mui/material';

interface ColorModeValue { mode: PaletteMode; toggleColorMode: () => void }
const ColorModeContext = createContext<ColorModeValue | undefined>(undefined);
const STORAGE_KEY = 'aigateway.console.color-mode';

export function ColorModeProvider({ children }: PropsWithChildren) {
  const [mode, setMode] = useState<PaletteMode>(() => localStorage.getItem(STORAGE_KEY) === 'dark' ? 'dark' : 'light');
  const value = useMemo<ColorModeValue>(() => ({ mode, toggleColorMode: () => setMode((current) => {
    const next = current === 'light' ? 'dark' : 'light'; localStorage.setItem(STORAGE_KEY, next); return next;
  }) }), [mode]);
  const theme = useMemo(() => createTheme({
    palette: { mode, primary: { main: mode === 'light' ? '#2457d6' : '#91adff' }, secondary: { main: '#168f79' },
      background: mode === 'light' ? { default: '#f5f7fa', paper: '#ffffff' } : { default: '#10141c', paper: '#171d27' },
      divider: mode === 'light' ? '#e4e9f0' : '#2a3341' },
    typography: { fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      h1: { fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.035em' }, h2: { fontSize: '1.5rem', fontWeight: 700 },
      h3: { fontSize: '1.125rem', fontWeight: 650 }, body2: { lineHeight: 1.55 }, button: { textTransform: 'none', fontWeight: 600 } },
    shape: { borderRadius: 12 },
    components: { MuiPaper: { styleOverrides: { root: { backgroundImage: 'none', border: `1px solid ${mode === 'light' ? '#e4e9f0' : '#2a3341'}` } } },
      MuiCard: { styleOverrides: { root: { boxShadow: 'none' } } }, MuiButton: { defaultProps: { disableElevation: true } } },
  }), [mode]);
  return <ColorModeContext.Provider value={value}><ThemeProvider theme={theme}><CssBaseline />{children}</ThemeProvider></ColorModeContext.Provider>;
}
export function useColorMode(): ColorModeValue {
  const context = useContext(ColorModeContext); if (!context) throw new Error('useColorMode must be used inside ColorModeProvider'); return context;
}
