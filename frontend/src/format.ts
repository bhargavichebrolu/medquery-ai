// Presentation-only formatting. The values in src/data stay exact; these helpers
// only decide how many decimal places to show on screen.

export function formatDecimal(value: number, decimals: number): string {
  return value.toFixed(decimals)
}

const ORDINALS = ['1st', '2nd', '3rd', '4th', '5th']

// A reciprocal rank of 1/n means the relevant article first appeared at position n.
// 0 means it was not in the top 5.
export function rankLabel(reciprocalRank: number): string {
  if (reciprocalRank === 0) {
    return 'Not in top 5'
  }
  return ORDINALS[Math.round(1 / reciprocalRank) - 1]
}

// A duration: milliseconds below one second, seconds above.
export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(1)} ms`
}
