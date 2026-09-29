import { afterEach, describe, expect, it, vi } from 'vitest';
import { formatMoney, formatNumber, formatTimestamp, formatWindow, windowStart } from './format';

afterEach(() => {
  vi.useRealTimers();
});

describe('formatMoney', () => {
  it('uses more precision for sub-cent amounts', () => {
    expect(formatMoney(12.5)).toMatch(/12\.50/);
    expect(formatMoney(0.000123)).toMatch(/0\.000123/);
    expect(formatMoney(0)).toMatch(/0\.00/);
  });

  it.each([null, undefined, Number.NaN])('reports %s as not available', (value) => {
    expect(formatMoney(value)).toBe('Not available');
  });
});

describe('formatNumber', () => {
  it('formats with the requested precision', () => {
    expect(formatNumber(1234.567, 1)).toMatch(/1.?234\.6/);
    expect(formatNumber(2)).toBe('2');
    expect(formatNumber(null)).toBe('Not available');
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe('Not available');
  });
});

describe('formatTimestamp', () => {
  it('describes past times relative to now', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-29T12:00:00Z'));
    expect(formatTimestamp('2026-09-29T11:00:00Z').relative).toMatch(/hour/);
    expect(formatTimestamp('2026-09-29T12:00:00Z').relative).toMatch(/now|0 seconds/);
    expect(formatTimestamp('2025-09-29T12:00:00Z').relative).toMatch(/year/);
    expect(formatTimestamp('2026-09-29T11:00:00Z').absolute).toMatch(/2026/);
  });

  it('keeps unparseable input as-is', () => {
    expect(formatTimestamp('not a date')).toEqual({ absolute: 'not a date', relative: 'Unknown time' });
  });
});

describe('windows', () => {
  it('names known windows and passes unknown ones through', () => {
    expect(formatWindow('24h')).toBe('last 24 hours');
    expect(formatWindow('90d')).toBe('90d');
  });

  it('computes window start times', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-29T12:00:00Z'));
    expect(windowStart('1h')).toBe('2026-09-29T11:00:00.000Z');
    expect(windowStart('forever')).toBeUndefined();
  });
});
