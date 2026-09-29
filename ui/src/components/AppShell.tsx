import { useState } from 'react';
import { AppBar, Avatar, Box, Divider, Drawer, IconButton, List, ListItemButton, ListItemIcon, ListItemText, Menu, MenuItem, Stack, Toolbar, Tooltip, Typography } from '@mui/material';
import DashboardRounded from '@mui/icons-material/DashboardRounded';
import KeyRounded from '@mui/icons-material/KeyRounded';
import SmartToyRounded from '@mui/icons-material/SmartToyRounded';
import ReceiptLongRounded from '@mui/icons-material/ReceiptLongRounded';
import InsightsRounded from '@mui/icons-material/InsightsRounded';
import SecurityRounded from '@mui/icons-material/SecurityRounded';
import StorageRounded from '@mui/icons-material/StorageRounded';
import HubRounded from '@mui/icons-material/HubRounded';
import ExtensionRounded from '@mui/icons-material/ExtensionRounded';
import TerminalRounded from '@mui/icons-material/TerminalRounded';
import SettingsRounded from '@mui/icons-material/SettingsRounded';
import MenuOpenRounded from '@mui/icons-material/MenuOpenRounded';
import MenuRounded from '@mui/icons-material/MenuRounded';
import DarkModeRounded from '@mui/icons-material/DarkModeRounded';
import LightModeRounded from '@mui/icons-material/LightModeRounded';
import LogoutRounded from '@mui/icons-material/LogoutRounded';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthProvider';
import { useColorMode } from '../theme/ColorModeContext';

const drawerWidth = 252;
const nav = [
  { label: 'Dashboard', path: '/', icon: DashboardRounded },
  { label: 'Keys', path: '/keys', icon: KeyRounded, admin: true },
  { label: 'Models', path: '/models', icon: SmartToyRounded },
  { label: 'Logs', path: '/logs', icon: ReceiptLongRounded },
  { label: 'Usage', path: '/usage', icon: InsightsRounded },
  { label: 'Guardrails', path: '/guardrails', icon: SecurityRounded },
  { label: 'Cache', path: '/cache', icon: StorageRounded },
  { label: 'RAG', path: '/rag', icon: HubRounded },
  { label: 'MCP', path: '/mcp', icon: ExtensionRounded },
  { label: 'Playground', path: '/playground', icon: TerminalRounded, admin: true },
  { label: 'Settings', path: '/settings', icon: SettingsRounded, admin: true },
];

export function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const { user, logout } = useAuth(); const { mode, toggleColorMode } = useColorMode(); const navigate = useNavigate();
  const width = collapsed ? 72 : drawerWidth;
  return <Box sx={{ display: 'flex', minHeight: '100vh' }}>
    <AppBar position="fixed" color="inherit" elevation={0} sx={{ zIndex: (theme) => theme.zIndex.drawer + 1, border: 0, borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
      <Toolbar sx={{ minHeight: '68px !important', pl: `${width + 24}px !important`, transition: 'padding .18s ease' }}>
        <Typography variant="h3" sx={{ flexGrow: 1, color: 'text.primary' }}>AI Gateway <Typography component="span" color="text.secondary" variant="body2">Console</Typography></Typography>
        <Tooltip title={`Switch to ${mode === 'light' ? 'dark' : 'light'} mode`}><IconButton onClick={toggleColorMode}>{mode === 'light' ? <DarkModeRounded /> : <LightModeRounded />}</IconButton></Tooltip>
        <IconButton onClick={(event) => setMenuAnchor(event.currentTarget)} aria-label="User menu" sx={{ ml: 1 }}><Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main', fontSize: 13 }}>{user?.username.slice(0, 1).toUpperCase()}</Avatar></IconButton>
        <Menu anchorEl={menuAnchor} open={Boolean(menuAnchor)} onClose={() => setMenuAnchor(null)}>
          <MenuItem disabled>{user?.username} · {user?.role}</MenuItem><Divider />
          <MenuItem onClick={() => { setMenuAnchor(null); logout(); navigate('/login', { replace: true }); }}><LogoutRounded fontSize="small" sx={{ mr: 1 }} />Sign out</MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
    <Drawer variant="permanent" sx={{ width, flexShrink: 0, '& .MuiDrawer-paper': { width, boxSizing: 'border-box', borderTop: 0, borderBottom: 0, borderLeft: 0, transition: 'width .18s ease', overflowX: 'hidden' } }}>
      <Toolbar sx={{ minHeight: '68px !important' }} />
      <Stack direction="row" justifyContent={collapsed ? 'center' : 'flex-end'} sx={{ px: 1.5, py: 1 }}>
        <Tooltip title={collapsed ? 'Expand navigation' : 'Collapse navigation'}><IconButton size="small" onClick={() => setCollapsed((value) => !value)}>{collapsed ? <MenuRounded /> : <MenuOpenRounded />}</IconButton></Tooltip>
      </Stack>
      <List sx={{ px: 1, pt: 0 }}>
        {nav.filter((item) => !item.admin || user?.role === 'admin').map(({ label, path, icon: Icon }) => <ListItemButton key={path} component={NavLink} to={path} end={path === '/'} title={collapsed ? label : undefined}
          sx={{ minHeight: 44, borderRadius: 2, mb: 0.5, '&.active': { bgcolor: 'action.selected', color: 'primary.main', '& .MuiListItemIcon-root': { color: 'primary.main' } } }}>
          <ListItemIcon sx={{ minWidth: collapsed ? 0 : 38, justifyContent: 'center' }}><Icon fontSize="small" /></ListItemIcon>{!collapsed && <ListItemText primary={label} primaryTypographyProps={{ variant: 'body2', fontWeight: 550 }} />}
        </ListItemButton>)}
      </List>
    </Drawer>
    <Box component="main" sx={{ flexGrow: 1, minWidth: 0, p: { xs: 2, md: 4 }, pt: { xs: 11, md: 12 } }}><Outlet /></Box>
  </Box>;
}
