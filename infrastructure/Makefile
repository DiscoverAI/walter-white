project_name = "sars-cov-2"
service_name = "$(project_name)-walter-white"
bucket_name = "$(project_name)-25309b4013524"

.PHONY: compute-environment-template-test
compute-environment-template-test:
	aws cloudformation validate-template --template-body file://./compute-environment.yaml >> /dev/null

.PHONY: compute-environment
compute-environment:
	aws cloudformation deploy \
		--template-file compute-environment.yaml \
		--stack-name "$(project_name)-batch-job-compute-environment" \
		--parameter-overrides ProjectName=$(project_name) DatalakeStackName="$(project_name)-datalake"

.PHONY: roles-template-test
roles-template-test:
	aws cloudformation validate-template --template-body file://./roles.yaml >> /dev/null

.PHONY: roles
roles:
	aws cloudformation deploy \
		--template-file roles.yaml \
		--stack-name "$(service_name)-roles" \
		--capabilities CAPABILITY_IAM \
		--parameter-overrides ProjectName=$(project_name) DatalakeStackName="$(project_name)-datalake"

.PHONY: infrastructure
infrastructure: roles
