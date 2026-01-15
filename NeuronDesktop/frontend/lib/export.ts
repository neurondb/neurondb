export type ExportFormat = 'csv' | 'json' | 'pdf'

export interface ExportOptions {
  format: ExportFormat
  filename?: string
  includeHeaders?: boolean
}

export function exportToCSV(data: any[], filename = 'export.csv') {
  if (data.length === 0) return

  const headers = Object.keys(data[0])
  const csvContent = [
    headers.join(','),
    ...data.map((row) =>
      headers
        .map((header) => {
          const value = row[header]
          return typeof value === 'string' && value.includes(',')
            ? `"${value.replace(/"/g, '""')}"`
            : value
        })
        .join(',')
    ),
  ].join('\n')

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = filename
  link.click()
}

export function exportToJSON(data: any[], filename = 'export.json') {
  const jsonContent = JSON.stringify(data, null, 2)
  const blob = new Blob([jsonContent], { type: 'application/json' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = filename
  link.click()
}

export function exportToPDF(data: any[], filename = 'export.pdf') {
  // For PDF export, we'll use a simple HTML to PDF approach
  // In production, you might want to use a library like jsPDF or pdfmake
  const htmlContent = `
    <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; padding: 20px; }
          table { width: 100%; border-collapse: collapse; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
        </style>
      </head>
      <body>
        <h1>Export Data</h1>
        <table>
          <thead>
            <tr>
              ${Object.keys(data[0] || {}).map((key) => `<th>${key}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${data.map((row) => `
              <tr>
                ${Object.values(row).map((val) => `<td>${val}</td>`).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </body>
    </html>
  `

  const blob = new Blob([htmlContent], { type: 'text/html' })
  const url = URL.createObjectURL(blob)
  window.open(url, '_blank')
  
  // Note: This is a simplified PDF export. For production, use a proper PDF library
  console.warn('PDF export opened in new window. For proper PDF generation, use a library like jsPDF.')
}

export function exportData(data: any[], options: ExportOptions) {
  const filename = options.filename || `export.${options.format}`

  switch (options.format) {
    case 'csv':
      exportToCSV(data, filename)
      break
    case 'json':
      exportToJSON(data, filename)
      break
    case 'pdf':
      exportToPDF(data, filename)
      break
    default:
      throw new Error(`Unsupported export format: ${options.format}`)
  }
}


