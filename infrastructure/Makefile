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
		--stack-name "$(project_name)-compute-environment" \
		--capabilities CAPABILITY_IAM \
		--parameter-overrides ProjectName=$(project_name) DatalakeStackName="$(project_name)-datalake"

.PHONY: walter-white-job-template-test
walter-white-job-template-test:
	aws cloudformation validate-template --template-body file://./walter-white-job.yaml >> /dev/null

.PHONY: walter-white-job
walter-white-job:
	aws cloudformation deploy \
		--template-file walter-white-job.yaml \
		--stack-name "$(service_name)" \
		--parameter-overrides ServiceName=$(service_name) BucketUri="s3://$(bucket_name)"

.PHONY: infrastructure
infrastructure: compute-environment walter-white-job

.PHONY: template-test
template-test: compute-environment-template-test walter-white-job-template-test
