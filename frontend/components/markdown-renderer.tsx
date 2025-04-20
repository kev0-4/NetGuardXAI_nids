"use client"

import { useEffect, useState } from "react"

interface MarkdownRendererProps {
  content: string
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const [formattedContent, setFormattedContent] = useState("")

  useEffect(() => {
    if (!content) return

    // Format bold text
    let formatted = content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")

    // Format italic text
    formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>")

    // Format lists (simple implementation)
    formatted = formatted.replace(/^\* (.*?)$/gm, "<li>$1</li>")
    formatted = formatted.replace(/<li>(.*?)<\/li>(\n<li>.*?<\/li>)+/g, "<ul>$&</ul>")

    // Format paragraphs
    formatted = formatted.replace(/\n\n/g, "</p><p>")

    // Wrap in paragraph if not already
    if (!formatted.startsWith("<")) {
      formatted = `<p>${formatted}</p>`
    }

    setFormattedContent(formatted)
  }, [content])

  return <div className="markdown-content text-gray-300" dangerouslySetInnerHTML={{ __html: formattedContent }} />
}
