import { matchPath } from 'react-router'

// One description per page. The sidebar, header and placeholder pages all read
// from here, so a page's address and name are defined in exactly one place.
// Pages that are not built yet name the milestone that builds them in plannedIn.
// Pages with child addresses (such as /knowledge-base/asthma) set hasChildPages.
export interface PageInfo {
  path: string
  label: string
  purpose: string
  plannedIn?: string
  hasChildPages?: boolean
}

export const PAGES = {
  dashboard: {
    path: '/',
    label: 'Dashboard',
    purpose: 'An overview of the knowledge base, the selected embedding model and recent experiments.',
  },
  chat: {
    path: '/chat',
    label: 'Chat',
    purpose:
      'Ask questions about the NHS documents and receive answers with citations to the retrieved evidence.',
    plannedIn: 'Part 3',
  },
  retrieval: {
    path: '/retrieval',
    label: 'Retrieval',
    purpose:
      'Ask a question and see the chunks an embedding model retrieves from the research knowledge base, with similarity scores.',
  },
  knowledgeBase: {
    path: '/knowledge-base',
    label: 'Knowledge Base',
    purpose: 'Browse the source documents, their metadata and the chunks prepared for retrieval.',
    hasChildPages: true,
  },
  modelLab: {
    path: '/model-lab',
    label: 'Model Lab',
    purpose: 'Choose an embedding model (MPNet, MiniLM, BGE or E5) and view its configuration.',
  },
  analytics: {
    path: '/analytics',
    label: 'Analytics',
    purpose: 'Research and evaluation results comparing the embedding models.',
  },
} satisfies Record<string, PageInfo>

// The order the pages appear in the sidebar.
export const NAV_ITEMS: PageInfo[] = [
  PAGES.dashboard,
  PAGES.chat,
  PAGES.retrieval,
  PAGES.knowledgeBase,
  PAGES.modelLab,
  PAGES.analytics,
]

// Find the page whose address matches the browser's current path, if any.
// A page with child pages also matches the addresses beneath it.
export function findPage(pathname: string): PageInfo | undefined {
  return NAV_ITEMS.find((page) =>
    matchPath({ path: page.path, end: !page.hasChildPages }, pathname),
  )
}
