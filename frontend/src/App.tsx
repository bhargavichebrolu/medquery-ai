import { Route, Routes } from 'react-router'
import AppLayout from './components/AppLayout'
import { PAGES } from './navigation'
import AnalyticsPage from './pages/AnalyticsPage'
import ArticleDetailPage from './pages/ArticleDetailPage'
import ChatPage from './pages/ChatPage'
import DashboardPage from './pages/DashboardPage'
import KnowledgeBasePage from './pages/KnowledgeBasePage'
import ModelLabPage from './pages/ModelLabPage'
import NotFoundPage from './pages/NotFoundPage'
import RetrievalPage from './pages/RetrievalPage'
import ModelSelectionProvider from './state/ModelSelectionProvider'

// Which page to show for each address. Every route sits inside AppLayout,
// so all pages share the same sidebar and header. The provider around them
// lets any page read the selected embedding model.
function App() {
  return (
    <ModelSelectionProvider>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path={PAGES.dashboard.path} element={<DashboardPage />} />
          <Route path={PAGES.chat.path} element={<ChatPage />} />
          <Route path={PAGES.retrieval.path} element={<RetrievalPage />} />
          <Route path={PAGES.knowledgeBase.path} element={<KnowledgeBasePage />} />
          {/* :articleId is a route parameter: /knowledge-base/asthma gives articleId = "asthma". */}
          <Route path={`${PAGES.knowledgeBase.path}/:articleId`} element={<ArticleDetailPage />} />
          <Route path={PAGES.modelLab.path} element={<ModelLabPage />} />
          <Route path={PAGES.analytics.path} element={<AnalyticsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </ModelSelectionProvider>
  )
}

export default App
