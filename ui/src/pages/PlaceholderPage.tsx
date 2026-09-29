import { useEffect } from 'react';
import { EmptyState, PageHeader } from '../components/Shared';
export default function PlaceholderPage({ title, description }: { title: string; description: string }) {
  useEffect(() => { document.title = `${title} · AI Gateway Console`; }, [title]);
  return <><PageHeader title={title} description={description} /><EmptyState title={`${title} is coming soon`} description="This console area is being prepared. Its API integration will be added in a follow-up." /></>;
}
