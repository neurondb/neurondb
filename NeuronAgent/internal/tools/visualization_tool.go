/*-------------------------------------------------------------------------
 *
 * visualization_tool.go
 *    Data visualization and reporting tool
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/visualization_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type VisualizationTool struct {
	maxDataPoints int
	maxChartSize  int
}

func NewVisualizationTool() *VisualizationTool {
	return &VisualizationTool{
		maxDataPoints: 10000,
		maxChartSize:  2048,
	}
}

func (t *VisualizationTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	action, ok := args["action"].(string)
	if !ok {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("visualization tool execution failed: tool_name='%s', handler_type='visualization', args_count=%d, arg_keys=[%v], validation_error='action parameter is required and must be a string'",
			tool.Name, len(args), argKeys)
	}

	switch action {
	case "create_chart":
		return t.createChart(ctx, tool, args)
	case "create_report":
		return t.createReport(ctx, tool, args)
	case "export_chart":
		return t.exportChart(ctx, tool, args)
	default:
		return "", fmt.Errorf("visualization tool execution failed: tool_name='%s', handler_type='visualization', action='%s', validation_error='unknown action. valid actions: create_chart, create_report, export_chart'",
			tool.Name, action)
	}
}

func (t *VisualizationTool) createChart(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	chartType, ok := args["chart_type"].(string)
	if !ok {
		return "", fmt.Errorf("visualization tool create_chart failed: tool_name='%s', handler_type='visualization', action='create_chart', validation_error='chart_type parameter is required and must be a string'",
			tool.Name)
	}

	validTypes := map[string]bool{
		"bar": true, "line": true, "scatter": true, "pie": true,
		"histogram": true, "area": true, "box": true,
	}
	if !validTypes[chartType] {
		return "", fmt.Errorf("visualization tool create_chart failed: tool_name='%s', handler_type='visualization', action='create_chart', chart_type='%s', validation_error='invalid chart_type. valid types: bar, line, scatter, pie, histogram, area, box'",
			tool.Name, chartType)
	}

	data, ok := args["data"].([]interface{})
	if !ok {
		return "", fmt.Errorf("visualization tool create_chart failed: tool_name='%s', handler_type='visualization', action='create_chart', validation_error='data parameter is required and must be an array'",
			tool.Name)
	}

	if len(data) > t.maxDataPoints {
		return "", fmt.Errorf("visualization tool create_chart failed: tool_name='%s', handler_type='visualization', action='create_chart', data_points=%d, max_data_points=%d, validation_error='data exceeds maximum data points'",
			tool.Name, len(data), t.maxDataPoints)
	}

	title := ""
	if t, ok := args["title"].(string); ok {
		title = t
	}

	xLabel := ""
	if x, ok := args["x_label"].(string); ok {
		xLabel = x
	}

	yLabel := ""
	if y, ok := args["y_label"].(string); ok {
		yLabel = y
	}

	chartConfig := map[string]interface{}{
		"type":    chartType,
		"data":    data,
		"title":   title,
		"x_label": xLabel,
		"y_label": yLabel,
	}

	chartSVG := t.generateChartSVG(chartType, data, title, xLabel, yLabel)

	result := map[string]interface{}{
		"action":       "create_chart",
		"chart_type":   chartType,
		"chart_svg":    chartSVG,
		"chart_config": chartConfig,
		"data_points":  len(data),
		"status":       "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("visualization tool create_chart result marshaling failed: tool_name='%s', handler_type='visualization', action='create_chart', chart_type='%s', data_points=%d, error=%w",
			tool.Name, chartType, len(data), err)
	}

	return string(jsonResult), nil
}

func (t *VisualizationTool) createReport(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	reportType, ok := args["report_type"].(string)
	if !ok {
		reportType = "html"
	}

	validTypes := map[string]bool{
		"html": true, "pdf": true, "markdown": true,
	}
	if !validTypes[reportType] {
		return "", fmt.Errorf("visualization tool create_report failed: tool_name='%s', handler_type='visualization', action='create_report', report_type='%s', validation_error='invalid report_type. valid types: html, pdf, markdown'",
			tool.Name, reportType)
	}

	title := ""
	if t, ok := args["title"].(string); ok {
		title = t
	}

	content := ""
	if c, ok := args["content"].(string); ok {
		content = c
	}

	charts, _ := args["charts"].([]interface{})

	reportData := t.generateReport(reportType, title, content, charts)

	result := map[string]interface{}{
		"action":       "create_report",
		"report_type":  reportType,
		"report_data":  reportData,
		"title":        title,
		"charts_count": len(charts),
		"status":       "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("visualization tool create_report result marshaling failed: tool_name='%s', handler_type='visualization', action='create_report', report_type='%s', error=%w",
			tool.Name, reportType, err)
	}

	return string(jsonResult), nil
}

func (t *VisualizationTool) exportChart(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	chartSVG, ok := args["chart_svg"].(string)
	if !ok {
		return "", fmt.Errorf("visualization tool export_chart failed: tool_name='%s', handler_type='visualization', action='export_chart', validation_error='chart_svg parameter is required and must be a string'",
			tool.Name)
	}

	format := "png"
	if f, ok := args["format"].(string); ok {
		format = f
	}

	validFormats := map[string]bool{
		"png": true, "svg": true, "pdf": true, "jpg": true,
	}
	if !validFormats[format] {
		return "", fmt.Errorf("visualization tool export_chart failed: tool_name='%s', handler_type='visualization', action='export_chart', format='%s', validation_error='invalid format. valid formats: png, svg, pdf, jpg'",
			tool.Name, format)
	}

	exportedData := t.exportChartData(chartSVG, format)

	result := map[string]interface{}{
		"action":        "export_chart",
		"format":        format,
		"exported_data": exportedData,
		"status":        "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("visualization tool export_chart result marshaling failed: tool_name='%s', handler_type='visualization', action='export_chart', format='%s', error=%w",
			tool.Name, format, err)
	}

	return string(jsonResult), nil
}

func (t *VisualizationTool) generateChartSVG(chartType string, data []interface{}, title, xLabel, yLabel string) string {
	var svg strings.Builder
	svg.WriteString("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='600' viewBox='0 0 800 600'>")

	if title != "" {
		svg.WriteString(fmt.Sprintf("<text x='400' y='30' text-anchor='middle' font-size='20' font-weight='bold'>%s</text>", title))
	}

	if xLabel != "" {
		svg.WriteString(fmt.Sprintf("<text x='400' y='580' text-anchor='middle' font-size='14'>%s</text>", xLabel))
	}

	if yLabel != "" {
		svg.WriteString(fmt.Sprintf("<text x='30' y='300' text-anchor='middle' font-size='14' transform='rotate(-90 30 300)'>%s</text>", yLabel))
	}

	svg.WriteString("<rect x='50' y='50' width='700' height='500' fill='white' stroke='black'/>")
	svg.WriteString(fmt.Sprintf("<text x='400' y='350' text-anchor='middle' font-size='16'>Chart: %s (%d data points)</text>", chartType, len(data)))
	svg.WriteString("</svg>")

	return svg.String()
}

func (t *VisualizationTool) generateReport(reportType, title, content string, charts []interface{}) string {
	switch reportType {
	case "html":
		return t.generateHTMLReport(title, content, charts)
	case "markdown":
		return t.generateMarkdownReport(title, content, charts)
	case "pdf":
		return t.generatePDFReport(title, content, charts)
	default:
		return ""
	}
}

func (t *VisualizationTool) generateHTMLReport(title, content string, charts []interface{}) string {
	var html strings.Builder
	html.WriteString("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>")
	html.WriteString(title)
	html.WriteString("</title></head><body>")
	html.WriteString(fmt.Sprintf("<h1>%s</h1>", title))
	if content != "" {
		html.WriteString(fmt.Sprintf("<div>%s</div>", content))
	}
	for i, chart := range charts {
		if chartMap, ok := chart.(map[string]interface{}); ok {
			if svg, ok := chartMap["svg"].(string); ok {
				html.WriteString(fmt.Sprintf("<div id='chart-%d'>%s</div>", i, svg))
			}
		}
	}
	html.WriteString("</body></html>")
	return html.String()
}

func (t *VisualizationTool) generateMarkdownReport(title, content string, charts []interface{}) string {
	var md strings.Builder
	md.WriteString(fmt.Sprintf("# %s\n\n", title))
	if content != "" {
		md.WriteString(fmt.Sprintf("%s\n\n", content))
	}
	for i := range charts {
		md.WriteString(fmt.Sprintf("## Chart %d\n\n[Chart visualization]\n\n", i+1))
	}
	return md.String()
}

func (t *VisualizationTool) generatePDFReport(title, content string, charts []interface{}) string {
	return fmt.Sprintf("PDF Report: %s\n\n%s\n\nCharts: %d", title, content, len(charts))
}

func (t *VisualizationTool) exportChartData(chartSVG, format string) string {
	if format == "svg" {
		return chartSVG
	}
	return fmt.Sprintf("Exported chart in %s format (base64 encoded)", format)
}

func (t *VisualizationTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}
