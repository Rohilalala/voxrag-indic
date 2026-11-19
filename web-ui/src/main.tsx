import { Component, StrictMode, type ErrorInfo, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'

import App from '@/App'
import './index.css'

/**
 * Last line of defence. A render-time exception anywhere in the tree would otherwise
 * unmount everything and leave a black screen mid-demo; this turns it into something a
 * person can read and recover from without a devtools console.
 */
class RootErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[voxrag] render failed', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="flex min-h-dvh items-center justify-center p-8">
        <div className="max-w-lg space-y-3">
          <p className="text-xs font-mono tracking-widest text-muted-foreground uppercase">
            interface error
          </p>
          <h1 className="text-xl font-medium">The UI stopped rendering.</h1>
          <p className="text-sm text-muted-foreground">
            The pipeline itself is unaffected &mdash; the API is still answering at{' '}
            <code className="font-mono">/api/health</code>.
          </p>
          <pre className="max-h-48 overflow-auto rounded-md border border-border bg-card p-3 font-mono text-xs whitespace-pre-wrap text-destructive">
            {error.message}
          </pre>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground"
          >
            Reload
          </button>
        </div>
      </div>
    )
  }
}

const container = document.getElementById('root')
if (!container) throw new Error('#root is missing from index.html')

createRoot(container).render(
  <StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </StrictMode>,
)
