'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import rehypeHighlight from 'rehype-highlight'

interface MarkdownContentProps {
  content: string
  className?: string
}

export default function MarkdownContent({ content, className = '' }: MarkdownContentProps) {
  // Convert literal \n strings to actual newlines
  // This handles cases where the AI response has escaped newlines
  const processedContent = content.replace(/\\n/g, '\n')
  
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize, rehypeHighlight]}
        components={{
          // Customize heading styles
          h1: ({ node, ...props }) => (
            <h1 className="text-2xl font-bold mt-6 mb-4 text-slate-100" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="text-xl font-bold mt-5 mb-3 text-slate-100" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="text-lg font-bold mt-4 mb-2 text-slate-200" {...props} />
          ),
          h4: ({ node, ...props }) => (
            <h4 className="text-base font-bold mt-3 mb-2 text-slate-200" {...props} />
          ),
          h5: ({ node, ...props }) => (
            <h5 className="text-sm font-bold mt-2 mb-1 text-slate-300" {...props} />
          ),
          h6: ({ node, ...props }) => (
            <h6 className="text-xs font-bold mt-2 mb-1 text-slate-300" {...props} />
          ),
          // Customize paragraph styles
          p: ({ node, ...props }) => (
            <p className="mb-3 leading-relaxed" {...props} />
          ),
          // Customize list styles
          ul: ({ node, ...props }) => (
            <ul className="list-disc list-inside mb-3 space-y-1 pl-4" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="list-decimal list-inside mb-3 space-y-1 pl-4" {...props} />
          ),
          li: ({ node, ...props }) => (
            <li className="leading-relaxed" {...props} />
          ),
          // Customize blockquote styles
          blockquote: ({ node, ...props }) => (
            <blockquote
              className="border-l-4 border-slate-600 pl-4 my-3 italic text-slate-300"
              {...props}
            />
          ),
          // Customize code block styles
          code: ({ node, inline, className, children, ...props }: any) => {
            if (inline) {
              return (
                <code
                  className="bg-slate-900 text-pink-400 px-1.5 py-0.5 rounded text-sm font-mono"
                  {...props}
                >
                  {children}
                </code>
              )
            }
            return (
              <code
                className={`${className || ''} block bg-slate-900 p-3 rounded-md my-2 overflow-x-auto text-sm`}
                {...props}
              >
                {children}
              </code>
            )
          },
          pre: ({ node, ...props }) => (
            <pre
              className="bg-slate-900 border border-slate-700 rounded-md my-3 overflow-x-auto"
              {...props}
            />
          ),
          // Customize link styles
          a: ({ node, ...props }) => (
            <a
              className="text-blue-400 hover:text-blue-300 underline"
              target="_blank"
              rel="noopener noreferrer"
              {...props}
            />
          ),
          // Customize table styles
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto my-3">
              <table className="min-w-full border border-slate-700 rounded-md" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => (
            <thead className="bg-slate-800" {...props} />
          ),
          tbody: ({ node, ...props }) => (
            <tbody className="divide-y divide-slate-700" {...props} />
          ),
          tr: ({ node, ...props }) => (
            <tr className="border-b border-slate-700" {...props} />
          ),
          th: ({ node, ...props }) => (
            <th className="px-4 py-2 text-left text-slate-200 font-semibold" {...props} />
          ),
          td: ({ node, ...props }) => (
            <td className="px-4 py-2 text-slate-300" {...props} />
          ),
          // Customize horizontal rule
          hr: ({ node, ...props }) => (
            <hr className="my-4 border-slate-700" {...props} />
          ),
          // Customize strong/bold text
          strong: ({ node, ...props }) => (
            <strong className="font-bold text-slate-100" {...props} />
          ),
          // Customize emphasis/italic text
          em: ({ node, ...props }) => (
            <em className="italic text-slate-200" {...props} />
          ),
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  )
}

