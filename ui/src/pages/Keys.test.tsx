import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { VirtualKey } from '../api/types';
import Keys, { toLocalInput } from './Keys';

const createKey = vi.fn((_body: unknown, options: { onSuccess: (result: { key: string }) => void }) => options.onSuccess({ key: 'sk-aigw-first-key' }));
const updateKey = vi.fn((_args: unknown, options: { onSuccess: () => void }) => options.onSuccess());
let keyItems: VirtualKey[] = [];

vi.mock('../auth/AuthProvider', () => ({ useAuth: () => ({ user: { role: 'admin' } }) }));
vi.mock('../api/hooks/useKeys', () => ({ useKeys: () => ({ isLoading: false, isError: false, isFetching: false, data: { items: keyItems, total: keyItems.length } }) }));
vi.mock('../api/hooks/useModelsAdmin', () => ({ useAdminModels: () => ({ isLoading: false, data: { items: [{ name: 'gpt-4o' }, { name: 'llama3.1' }] } }) }));
vi.mock('../api/hooks/useTeams', () => ({ useTeams: () => ({ isLoading: false, data: { items: [{ id: 't1', name: 'Research' }] } }) }));
vi.mock('../api/hooks/useMcpServers', () => ({ useMcpServers: () => ({ data: [] }) }));
vi.mock('../api/hooks/useGuardrails', () => ({ useGuardrailPolicies: () => ({ data: { items: [{ name: 'default' }, { name: 'strict' }] } }) }));
vi.mock('../api/hooks/useKeysAdmin', () => ({
  useCreateKey: () => ({ mutate: createKey, isPending: false }),
  useUpdateKey: () => ({ mutate: updateKey, isPending: false }),
  useDeleteKey: () => ({ mutate: vi.fn(), isPending: false }),
  useRegenerateKey: () => ({ mutate: vi.fn(), isPending: false }),
}));

const existing: VirtualKey = {
  id: 'k1', key_prefix: 'sk-aigw-abcd', name: 'research-agent', team_id: 't1', max_budget_usd: 50, spend_usd: 1.25,
  budget_duration: 'weekly', rpm_limit: 60, tpm_limit: null, max_parallel_requests: 4, allowed_models: ['gpt-4o'], blocked_models: [],
  guardrail_policy: 'strict', allowed_routes: ['/v1/chat'], allowed_mcp_servers: [], allowed_tools: [], enabled: true,
  expires_at: '2030-01-02T03:04:00.000Z', last_used_at: null, metadata: {}, created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
} as VirtualKey;

beforeEach(() => {
  keyItems = [];
  createKey.mockClear();
  updateKey.mockClear();
});

describe('Keys page with no keys yet', () => {
  it('still opens the create dialog and shows the new secret', async () => {
    render(<Keys />);
    expect(screen.getByText('No virtual keys')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Create key' }));
    expect(await screen.findByRole('dialog', { name: 'Create virtual key' })).toBeInTheDocument();

    await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'first-app');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(createKey).toHaveBeenCalledWith(expect.objectContaining({ name: 'first-app', budget_duration: 'monthly', enabled: true }), expect.anything());
    expect(await screen.findByText('sk-aigw-first-key')).toBeInTheDocument();
  });

  it('allows all models by default, or only the models chosen', async () => {
    render(<Keys />);
    await userEvent.click(screen.getByRole('button', { name: 'Create key' }));
    await screen.findByRole('dialog', { name: 'Create virtual key' });
    const allModels = screen.getByRole('switch', { name: 'All models' });
    expect(allModels).toBeChecked();
    expect(screen.getByText('This key can use every model, including models added later.')).toBeInTheDocument();

    await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'scoped-app');
    await userEvent.click(allModels);
    expect(screen.getByText('Choose at least one model, or turn on All models.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();

    await userEvent.click(screen.getByRole('combobox', { name: 'Allowed models' }));
    await userEvent.click(await screen.findByRole('option', { name: 'gpt-4o' }));
    await userEvent.keyboard('{Escape}');
    expect(screen.getByText('1 model allowed')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(createKey).toHaveBeenCalledWith(expect.objectContaining({ name: 'scoped-app', allowed_models: ['gpt-4o'] }), expect.anything());
  });
});

describe('Editing a key', () => {
  it('shows every setting the key has and saves them all', async () => {
    keyItems = [existing];
    render(<Keys />);
    await userEvent.click(screen.getByRole('button', { name: 'Edit key' }));
    const dialog = await screen.findByRole('dialog', { name: 'Edit key “research-agent”' });
    const form = within(dialog);

    expect(form.getByRole('textbox', { name: 'Name' })).toHaveValue('research-agent');
    expect(form.getByRole('combobox', { name: 'Team' })).toHaveTextContent('Research');
    expect(form.getByRole('switch', { name: 'All models' })).not.toBeChecked();
    expect(form.getByRole('combobox', { name: 'Guardrail policy' })).toHaveTextContent('strict');
    expect(form.getByRole('textbox', { name: 'Allowed routes' })).toHaveValue('/v1/chat');
    expect(form.getByRole('combobox', { name: 'Budget period' })).toHaveTextContent('Weekly');
    expect(form.getByRole('spinbutton', { name: 'Max concurrent' })).toHaveValue(4);
    expect(form.getByLabelText('Expiry')).toHaveValue(toLocalInput(existing.expires_at));
    expect(form.getByRole('switch', { name: 'Enabled' })).toBeChecked();

    await userEvent.clear(form.getByRole('textbox', { name: 'Name' }));
    await userEvent.type(form.getByRole('textbox', { name: 'Name' }), 'research-agent-v2');
    await userEvent.click(form.getByRole('combobox', { name: 'Budget period' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Daily' }));
    await userEvent.click(form.getByRole('switch', { name: 'Enabled' }));
    await userEvent.click(form.getByRole('button', { name: 'Save changes' }));

    expect(updateKey).toHaveBeenCalledWith(
      {
        id: 'k1',
        body: expect.objectContaining({
          name: 'research-agent-v2', team_id: 't1', budget_duration: 'daily', enabled: false,
          allowed_models: ['gpt-4o'], guardrail_policy: 'strict', max_parallel_requests: 4, allowed_routes: ['/v1/chat'],
        }),
      },
      expect.anything(),
    );
  });
});

describe('toLocalInput', () => {
  it('formats an ISO timestamp for a datetime-local input in local time', () => {
    const iso = '2030-01-02T03:04:00.000Z';
    expect(new Date(toLocalInput(iso)).toISOString()).toBe(iso);
    expect(toLocalInput(null)).toBe('');
    expect(toLocalInput('not a date')).toBe('');
  });
});
