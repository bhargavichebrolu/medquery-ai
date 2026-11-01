import { matchPath } from 'react-router'

// One description per page. The sidebar, header and placeholder pages all read
// from here, so a page's address and name are defined in exactly one place.
export interface PageInfo {
  path: string
  label: string
  purpose: string
  plannedIn: string
}

export const PAGES = {
  dashboard: {
    path: '/',
    label: 'Dashboard',
    purpose: 'An overview of the knowledge base, the selected embedding model and recent experiments.',
    plannedIn: 'PRD 1.3',
  },
  chat: {
    path: '/chat',
    label: 'Chat',
    purpose:
      'Ask questions about the NHS documents and receive answers with citations to the retrieved evidence.',
    plannedIn: 'Part 3',
  },
  knowledgeBase: {
    path: '/knowledge-base',
    label: 'Knowledge Base',
    purpose: 'Browse the source documents, their metadata and the chunks prepared for retrieval.',
    plannedIn: 'PRD 1.6',
  },
  modelLab: {
    path: '/model-lab',
    label: 'Model Lab',
    purpose: 'Choose an embedding model (MPNet, MiniLM, BGE or E5) and view its configuration.',
    plannedIn: 'PRD 1.4',
  },
  analytics: {
    path: '/analytics',
    label: 'Analytics',
    purpose: 'Research and evaluation results comparing the embedding models.',
    plannedIn: 'PRD 1.5',
  },
} satisfies Record<string, PageInfo>

// The order the pages appear in the sidebar.
export const NAV_ITEMS: PageInfo[] = [
  PAGES.dashboard,
  PAGES.chat,
  PAGES.knowledgeBase,
  PAGES.modelLab,
  PAGES.analytics,
]

// Find the page whose address matches the browser's current path, if any.
export function findPage(pathname: string): PageInfo | undefined {
  return NAV_ITEMS.find((page) => matchPath({ path: page.path, end: true }, pathname))
}
