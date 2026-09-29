{{- define "aigateway.name" -}}{{ default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}{{- end }}
{{- define "aigateway.fullname" -}}{{ default (include "aigateway.name" .) .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}{{- end }}
{{- define "aigateway.labels" -}}
app.kubernetes.io/name: {{ include "aigateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
{{- define "aigateway.serviceAccountName" -}}{{ if .Values.serviceAccount.create }}{{ default (include "aigateway.fullname" .) .Values.serviceAccount.name }}{{ else }}{{ default "default" .Values.serviceAccount.name }}{{ end }}{{- end }}
