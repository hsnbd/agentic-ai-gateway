{{- define "aigateway.name" -}}{{ default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}{{- end }}
{{- define "aigateway.fullname" -}}{{ default (include "aigateway.name" .) .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}{{- end }}
{{- define "aigateway.labels" -}}
app.kubernetes.io/name: {{ include "aigateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
{{- define "aigateway.serviceAccountName" -}}{{ if .Values.serviceAccount.create }}{{ default (include "aigateway.fullname" .) .Values.serviceAccount.name }}{{ else }}{{ default "default" .Values.serviceAccount.name }}{{ end }}{{- end }}

{{/* Container environment, shared by the gateway and the migration init container. */}}
{{- define "aigateway.env" -}}
{{- range $key, $value := .Values.env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- range $key, $_ := .Values.secretEnv }}
- name: {{ $key }}
  valueFrom: {secretKeyRef: {name: {{ include "aigateway.fullname" $ }}-secrets, key: {{ $key }}}}
{{- end }}
{{- if .Values.external.postgres.url }}
- name: DATABASE_URL
  value: {{ .Values.external.postgres.url | quote }}
{{- end }}
{{- if .Values.external.redis.url }}
- name: REDIS_URL
  value: {{ .Values.external.redis.url | quote }}
{{- end }}
{{- if .Values.migrations.enabled }}
- name: AUTO_CREATE_SCHEMA
  value: "false"
{{- end }}
{{- end }}
