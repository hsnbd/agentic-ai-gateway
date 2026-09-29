import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { LongText } from '../features/knowledge/LongText';
import { ColorModeProvider, useColorMode } from '../theme/ColorModeContext';
import { ErrorBoundary } from './ErrorBoundary';
import { ConfirmDialog, CopyButton, DataTable, EmptyState, ErrorState, JsonViewer, LoadingState, PageHeader, StatCard } from './Shared';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('ErrorBoundary', () => {
  function Broken(): never { throw new Error('render exploded'); }

  it('renders children until one throws, then offers a reload', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const assign = vi.fn();
    vi.stubGlobal('location', { assign });
    const { rerender } = render(<ErrorBoundary><p>fine</p></ErrorBoundary>);
    expect(screen.getByText('fine')).toBeInTheDocument();
    rerender(<ErrorBoundary><Broken /></ErrorBoundary>);
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('render exploded')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Reload console' }));
    expect(assign).toHaveBeenCalledWith('/ui/');
  });
});

describe('Shared primitives', () => {
  it('renders headers, stats, and empty/loading states with optional parts', () => {
    render(<>
      <PageHeader title="Keys" description="Virtual keys" action={<button>New</button>} />
      <PageHeader title="Bare" />
      <StatCard label="Requests" value="42" hint="last 24h" icon={<span>icon</span>} />
      <StatCard label="Cost" value="$1" />
      <EmptyState title="Nothing yet" description="Create one" />
      <EmptyState title="Custom" icon={<span>custom-icon</span>} />
      <LoadingState />
      <LoadingState label="Fetching logs" />
    </>);
    for (const text of ['Keys', 'Virtual keys', 'New', 'Bare', '42', 'last 24h', 'icon', 'Nothing yet', 'Create one', 'custom-icon', 'Loading', 'Fetching logs']) {
      expect(screen.getByText(text)).toBeInTheDocument();
    }
  });

  it('shows error messages and an optional retry', async () => {
    const retry = vi.fn();
    const { rerender } = render(<ErrorState error={new Error('nope')} onRetry={retry} />);
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(retry).toHaveBeenCalled();
    rerender(<ErrorState error="not an error" />);
    expect(screen.getByText('The request could not be completed.')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('confirms and cancels', async () => {
    const onClose = vi.fn();
    const onConfirm = vi.fn();
    const { rerender } = render(<ConfirmDialog open title="Delete key?" description="Gone for good" onClose={onClose} onConfirm={onConfirm} />);
    await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onConfirm).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
    rerender(<ConfirmDialog open title="Delete key?" description="Gone" confirmLabel="Delete" onClose={onClose} onConfirm={onConfirm} />);
    expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
  });

  it('copies values and resets its tooltip', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const writeText = vi.fn(() => Promise.resolve());
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    render(<><CopyButton value="sk-aigw-secret" /><CopyButton value="x" label="Copy id" /></>);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy' })); });
    expect(writeText).toHaveBeenCalledWith('sk-aigw-secret');
    await act(async () => { vi.advanceTimersByTime(1500); });
    expect(screen.getByRole('button', { name: 'Copy id' })).toBeInTheDocument();
  });

  it('renders JSON, including values JSON cannot represent', () => {
    const { container, rerender } = render(<JsonViewer value={{ a: 1 }} />);
    expect(container.textContent).toContain('"a": 1');
    rerender(<JsonViewer value={undefined} />);
    expect(container.textContent).toBe('null');
  });

  it('renders a data grid', () => {
    render(<DataTable rows={[{ id: 1, name: 'alpha' }]} columns={[{ field: 'name', headerName: 'Name' }]} />);
    expect(screen.getByRole('grid')).toBeInTheDocument();
  });
});

describe('LongText', () => {
  it('only offers expansion for long text', async () => {
    const { rerender } = render(<LongText text="short" />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    rerender(<LongText text={'x'.repeat(300)} />);
    await userEvent.click(screen.getByRole('button', { name: 'Expand' }));
    await userEvent.click(screen.getByRole('button', { name: 'Show less' }));
    rerender(<LongText text={'a\nb\nc'} lines={2} />);
    expect(screen.getByRole('button', { name: 'Expand' })).toBeInTheDocument();
  });
});

describe('ColorModeProvider', () => {
  function Toggle() {
    const { mode, toggleColorMode } = useColorMode();
    return <button onClick={toggleColorMode}>{mode}</button>;
  }

  it('toggles and remembers the mode', async () => {
    const { unmount } = render(<ColorModeProvider><Toggle /></ColorModeProvider>);
    await userEvent.click(screen.getByRole('button', { name: 'light' }));
    expect(screen.getByRole('button', { name: 'dark' })).toBeInTheDocument();
    unmount();
    render(<ColorModeProvider><Toggle /></ColorModeProvider>);
    await userEvent.click(screen.getByRole('button', { name: 'dark' }));
    expect(screen.getByRole('button', { name: 'light' })).toBeInTheDocument();
    expect(localStorage.getItem('aigateway.console.color-mode')).toBe('light');
  });

  it('refuses to be used outside the provider', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    expect(() => render(<Toggle />)).toThrow('useColorMode must be used inside ColorModeProvider');
  });
});
