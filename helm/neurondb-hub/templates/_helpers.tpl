{{/*
Expand the name of the chart.
*/}}
{{- define "neurondb-hub.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "neurondb-hub.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Full name (used by secrets and DNS)
*/}}
{{- define "neurondb-hub.fullname" -}}
{{- default (include "neurondb-hub.name" .) .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "neurondb-hub.labels" -}}
helm.sh/chart: {{ include "neurondb-hub.chart" . }}
app.kubernetes.io/name: {{ include "neurondb-hub.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{/*
Secret name (existing or chart-created)
*/}}
{{- define "neurondb-hub.secretName" -}}
{{- if .Values.secrets.existingSecretName }}
{{- .Values.secrets.existingSecretName }}
{{- else }}
{{- include "neurondb-hub.fullname" . }}-secrets
{{- end }}
{{- end }}

{{/*
Hub DB connection URL when hubDb.enabled and no secrets.databaseUrl
*/}}
{{- define "neurondb-hub.databaseUrl" -}}
{{- if .Values.secrets.databaseUrl }}
{{- .Values.secrets.databaseUrl }}
{{- else if .Values.hubDb.enabled }}
postgres://{{ .Values.hubDb.auth.username }}:{{ .Values.secrets.hubDbPassword | default .Values.hubDb.auth.password }}@{{ include "neurondb-hub.fullname" . }}-hub-db:{{ .Values.hubDb.port }}/{{ .Values.hubDb.auth.database }}?sslmode=disable
{{- else }}
{{- "" }}
{{- end }}
{{- end }}
