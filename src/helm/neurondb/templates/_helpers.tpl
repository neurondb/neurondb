{{/*
Expand the name of the chart.
*/}}
{{- define "neurondb.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "neurondb.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "neurondb.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "neurondb.labels" -}}
helm.sh/chart: {{ include "neurondb.chart" . }}
{{ include "neurondb.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "neurondb.selectorLabels" -}}
app.kubernetes.io/name: {{ include "neurondb.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
NeuronDB labels
*/}}
{{- define "neurondb.neurondb.labels" -}}
{{ include "neurondb.labels" . }}
app.kubernetes.io/component: neurondb
{{- end }}

{{/*
NeuronAgent labels
*/}}
{{- define "neurondb.neuronagent.labels" -}}
{{ include "neurondb.labels" . }}
app.kubernetes.io/component: neuronagent
{{- end }}

{{/*
NeuronMCP labels
*/}}
{{- define "neurondb.neuronmcp.labels" -}}
{{ include "neurondb.labels" . }}
app.kubernetes.io/component: neuronmcp
{{- end }}

{{/*
NeuronDesktop API labels
*/}}
{{- define "neurondb.neurondesktop-api.labels" -}}
{{ include "neurondb.labels" . }}
app.kubernetes.io/component: neurondesktop-api
{{- end }}

{{/*
NeuronDesktop Frontend labels
*/}}
{{- define "neurondb.neurondesktop-frontend.labels" -}}
{{ include "neurondb.labels" . }}
app.kubernetes.io/component: neurondesktop-frontend
{{- end }}

{{/*
NeuronDB selector labels
*/}}
{{- define "neurondb.neurondb.selectorLabels" -}}
{{ include "neurondb.selectorLabels" . }}
app.kubernetes.io/component: neurondb
{{- end }}

{{/*
NeuronAgent selector labels
*/}}
{{- define "neurondb.neuronagent.selectorLabels" -}}
{{ include "neurondb.selectorLabels" . }}
app.kubernetes.io/component: neuronagent
{{- end }}

{{/*
NeuronMCP selector labels
*/}}
{{- define "neurondb.neuronmcp.selectorLabels" -}}
{{ include "neurondb.selectorLabels" . }}
app.kubernetes.io/component: neuronmcp
{{- end }}

{{/*
NeuronDesktop API selector labels
*/}}
{{- define "neurondb.neurondesktop-api.selectorLabels" -}}
{{ include "neurondb.selectorLabels" . }}
app.kubernetes.io/component: neurondesktop-api
{{- end }}

{{/*
NeuronDesktop Frontend selector labels
*/}}
{{- define "neurondb.neurondesktop-frontend.selectorLabels" -}}
{{ include "neurondb.selectorLabels" . }}
app.kubernetes.io/component: neurondesktop-frontend
{{- end }}

{{/*
Create the name of the service account to use (legacy, for backward compatibility)
*/}}
{{- define "neurondb.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "neurondb.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
NeuronDB ServiceAccount name
*/}}
{{- define "neurondb.neurondb.serviceAccountName" -}}
{{- if .Values.rbac.enabled }}
{{- printf "%s-neurondb" (include "neurondb.fullname" .) }}
{{- else }}
{{- include "neurondb.serviceAccountName" . }}
{{- end }}
{{- end }}

{{/*
NeuronAgent ServiceAccount name
*/}}
{{- define "neurondb.neuronagent.serviceAccountName" -}}
{{- if .Values.rbac.enabled }}
{{- printf "%s-neuronagent" (include "neurondb.fullname" .) }}
{{- else }}
{{- include "neurondb.serviceAccountName" . }}
{{- end }}
{{- end }}

{{/*
NeuronMCP ServiceAccount name
*/}}
{{- define "neurondb.neuronmcp.serviceAccountName" -}}
{{- if .Values.rbac.enabled }}
{{- printf "%s-neuronmcp" (include "neurondb.fullname" .) }}
{{- else }}
{{- include "neurondb.serviceAccountName" . }}
{{- end }}
{{- end }}

{{/*
NeuronDesktop API ServiceAccount name
*/}}
{{- define "neurondb.neurondesktop-api.serviceAccountName" -}}
{{- if .Values.rbac.enabled }}
{{- printf "%s-neurondesktop-api" (include "neurondb.fullname" .) }}
{{- else }}
{{- include "neurondb.serviceAccountName" . }}
{{- end }}
{{- end }}

{{/*
NeuronDesktop Frontend ServiceAccount name
*/}}
{{- define "neurondb.neurondesktop-frontend.serviceAccountName" -}}
{{- if .Values.rbac.enabled }}
{{- printf "%s-neurondesktop-frontend" (include "neurondb.fullname" .) }}
{{- else }}
{{- include "neurondb.serviceAccountName" . }}
{{- end }}
{{- end }}

{{/*
NeuronDB service name
*/}}
{{- define "neurondb.neurondb.serviceName" -}}
{{- printf "%s-neurondb" (include "neurondb.fullname" .) }}
{{- end }}

{{/*
NeuronAgent service name
*/}}
{{- define "neurondb.neuronagent.serviceName" -}}
{{- printf "%s-neuronagent" (include "neurondb.fullname" .) }}
{{- end }}

{{/*
NeuronMCP service name
*/}}
{{- define "neurondb.neuronmcp.serviceName" -}}
{{- printf "%s-neuronmcp" (include "neurondb.fullname" .) }}
{{- end }}

{{/*
NeuronDesktop API service name
*/}}
{{- define "neurondb.neurondesktop-api.serviceName" -}}
{{- printf "%s-neurondesktop-api" (include "neurondb.fullname" .) }}
{{- end }}

{{/*
NeuronDesktop Frontend service name
*/}}
{{- define "neurondb.neurondesktop-frontend.serviceName" -}}
{{- printf "%s-neurondesktop-frontend" (include "neurondb.fullname" .) }}
{{- end }}

{{/*
Generate postgres password
*/}}
{{- define "neurondb.postgresPassword" -}}
{{- if .Values.secrets.postgresPassword }}
{{- .Values.secrets.postgresPassword }}
{{- else }}
{{- randAlphaNum 32 }}
{{- end }}
{{- end }}

{{/*
Get PostgreSQL host (internal or external). For CNPG, use -rw service (primary).
*/}}
{{- define "neurondb.postgresHost" -}}
{{- if .Values.neurondb.postgresql.external.enabled }}
{{- .Values.neurondb.postgresql.external.host | default "external-postgres" }}
{{- else if .Values.neurondb.postgresql.external.secretName }}
{{- .Values.neurondb.postgresql.external.host | default "external-postgres" }}
{{- else }}
{{- printf "%s-rw" (include "neurondb.neurondb.serviceName" .) }}
{{- end }}
{{- end }}

{{/*
Get PostgreSQL read-only host (CNPG -ro service). Only when not external.
*/}}
{{- define "neurondb.postgresHostReadOnly" -}}
{{- if .Values.neurondb.postgresql.external.enabled }}
{{- .Values.neurondb.postgresql.external.host | default "external-postgres" }}
{{- else }}
{{- printf "%s-ro" (include "neurondb.neurondb.serviceName" .) }}
{{- end }}
{{- end }}

{{/*
Get PostgreSQL read host (CNPG -r service). Only when not external.
*/}}
{{- define "neurondb.postgresHostRead" -}}
{{- if .Values.neurondb.postgresql.external.enabled }}
{{- .Values.neurondb.postgresql.external.host | default "external-postgres" }}
{{- else }}
{{- printf "%s-r" (include "neurondb.neurondb.serviceName" .) }}
{{- end }}
{{- end }}

{{/*
CNPG Cluster resource name (same as service base name).
*/}}
{{- define "neurondb.cnpgClusterName" -}}
{{- include "neurondb.neurondb.serviceName" . }}
{{- end }}

{{/*
Get PostgreSQL port (internal or external)
*/}}
{{- define "neurondb.postgresPort" -}}
{{- if .Values.neurondb.postgresql.external.enabled }}
{{- .Values.neurondb.postgresql.external.port | default 5432 }}
{{- else }}
{{- .Values.neurondb.postgresql.port }}
{{- end }}
{{- end }}

{{/*
Get PostgreSQL database name (internal or external)
*/}}
{{- define "neurondb.postgresDatabase" -}}
{{- if .Values.neurondb.postgresql.external.enabled }}
{{- .Values.neurondb.postgresql.external.database }}
{{- else }}
{{- .Values.neurondb.postgresql.database }}
{{- end }}
{{- end }}

{{/*
Get PostgreSQL username (internal or external)
*/}}
{{- define "neurondb.postgresUsername" -}}
{{- if .Values.neurondb.postgresql.external.enabled }}
{{- if .Values.neurondb.postgresql.external.secretName }}
{{- /* Username will come from secret */}}
{{- .Values.neurondb.postgresql.external.username | default "neurondb" }}
{{- else }}
{{- .Values.neurondb.postgresql.external.username }}
{{- end }}
{{- else }}
{{- .Values.neurondb.postgresql.username }}
{{- end }}
{{- end }}

{{/*
Get PostgreSQL password secret name and key
*/}}
{{- define "neurondb.postgresPasswordSecret" -}}
{{- if .Values.neurondb.postgresql.external.enabled }}
{{- if .Values.neurondb.postgresql.external.secretName }}
{{- .Values.neurondb.postgresql.external.secretName }}
{{- else }}
{{- include "neurondb.fullname" . }}-secrets
{{- end }}
{{- else }}
{{- include "neurondb.fullname" . }}-secrets
{{- end }}
{{- end }}

{{/*
Get PostgreSQL password secret key (CNPG uses "password" in basic-auth secret).
*/}}
{{- define "neurondb.postgresPasswordKey" -}}
{{- if and .Values.neurondb.postgresql.external.enabled .Values.neurondb.postgresql.external.secretName }}
{{- .Values.neurondb.postgresql.external.secretKeys.password | default "password" }}
{{- else }}
{{- "password" }}
{{- end }}
{{- end }}

{{/*
Fully qualified NeuronDB PostgreSQL container image (CUDA).

Naming:
  registry    — hostname only (ghcr.io, docker.io). Not the GHCR namespace.
  repository  — namespace/image matching Docker conventions: neurondb/neurondb-cuda
              (Docker Hub pull: docker pull neurondb/neurondb-cuda)
  tag         — release tag or latest / pg17-3.x.y etc.

Keeps neurondb.image independent of global.imageRegistry (used for Prometheus/Grafana paths).
*/}}
{{- define "neurondb.image.reference" -}}
{{- $repo := .Values.neurondb.image.repository | trim }}
{{- $tag := trim (default "latest" .Values.neurondb.image.tag) }}
{{- if or (hasPrefix "ghcr.io/" $repo) (hasPrefix "docker.io/" $repo) }}
{{- printf "%s:%s" $repo $tag }}
{{- else }}
{{- $reg := trim (default "ghcr.io" .Values.neurondb.image.registry) }}
{{- printf "%s/%s:%s" $reg (required "neurondb.image.repository must be set (use neurondb/neurondb-cuda)" $repo) $tag }}
{{- end }}
{{- end }}

