import { useState } from 'react';
import { Box, Button, Typography } from '@mui/material';

export function LongText({ text, lines = 4 }: { text: string; lines?: number }) {
  const [expanded, setExpanded] = useState(false);
  const clamped = text.length > 240 || text.split('\n').length > lines;
  return <Box sx={{ minWidth: 0, overflowWrap: 'anywhere' }}>
    <Typography
      variant="body2"
      sx={expanded ? { whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' } : {
        whiteSpace: 'pre-wrap', overflow: 'hidden', display: '-webkit-box', WebkitBoxOrient: 'vertical',
        WebkitLineClamp: lines, overflowWrap: 'anywhere',
      }}
    >{text}</Typography>
    {clamped && <Button size="small" onClick={() => setExpanded((value) => !value)} sx={{ px: 0, minWidth: 0 }}>
      {expanded ? 'Show less' : 'Expand'}
    </Button>}
  </Box>;
}
