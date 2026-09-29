const currencyFormat = (maximumFractionDigits: number) => new Intl.NumberFormat(undefined, {
  style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits,
});

export function formatMoney(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'Not available';
  const abs = Math.abs(value);
  return currencyFormat(abs > 0 && abs < 0.01 ? 6 : 2).format(value);
}

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'Not available';
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

export function formatTimestamp(value: string): { absolute: string; relative: string } {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return { absolute: value, relative: 'Unknown time' };
  const absolute = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' }).format(date);
  const seconds = (date.getTime() - Date.now()) / 1000;
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['year', 31_536_000], ['month', 2_592_000], ['week', 604_800], ['day', 86_400], ['hour', 3_600], ['minute', 60], ['second', 1],
  ];
  const [unit, duration] = units.find(([, size]) => Math.abs(seconds) >= size) ?? ['second', 1];
  return { absolute, relative: new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }).format(Math.round(seconds / duration), unit) };
}

export function formatWindow(window: string): string {
  return ({ '1h': 'last hour', '24h': 'last 24 hours', '7d': 'last 7 days', '30d': 'last 30 days' } as Record<string, string>)[window] ?? window;
}

export function windowStart(window: string): string | undefined {
  const durationMs: Record<string, number> = { '1h': 3_600_000, '24h': 86_400_000, '7d': 604_800_000, '30d': 2_592_000_000 };
  const duration = durationMs[window];
  return duration === undefined ? undefined : new Date(Date.now() - duration).toISOString();
}
