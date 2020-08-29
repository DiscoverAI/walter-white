project_name = "sars-cov-2"
module_name = "walter-white"
service_name = "$(project_name)-$(module_name)"
bucket_name = "$(project_name)-25309b4013524"

region = "eu-central-1"

admin_profile = "tortugas-administrator"
developer_profile = "tortugas-developer"

wait_for_status_change=5

now := $(shell date +%s)


.PHONY: compute-environment-template-test
compute-environment-template-test:
	aws cloudformation validate-template --profile $(developer_profile) --region $(region) --template-body file://./compute-environment.yaml >> /dev/null

.PHONY: compute-environment
compute-environment:
	aws cloudformation deploy \
		--template-file compute-environment.yaml \
		--stack-name "$(project_name)-compute-environment" \
		--region $(region) \
        --capabilities CAPABILITY_NAMED_IAM \
        --profile $(developer_profile) \
		--parameter-overrides ProjectName=$(project_name) DatalakeStackName="$(project_name)-datalake" VPCStackName=$(project_name)-vpc

.PHONY: update-compute-environment
update-compute-environment:
	aws cloudformation create-change-set \
		--change-set-name "$(project_name)-compute-environment-${now}" \
		--change-set-type UPDATE \
		--template-body file://./compute-environment.yaml \
		--stack-name "$(project_name)-compute-environment" \
		--region $(region) \
		--capabilities CAPABILITY_NAMED_IAM \
		--profile $(developer_profile) \
		--parameters ParameterKey=ProjectName,ParameterValue=$(project_name) ParameterKey=DatalakeStackName,ParameterValue="$(project_name)-datalake" ParameterKey=VPCStackName,ParameterValue="$(project_name)-vpc"

	sleep $(wait_for_status_change)

	aws cloudformation execute-change-set \
		--change-set-name "$(project_name)-compute-environment-${now}" \
		--stack-name "$(project_name)-compute-environment" \
		--region $(region) \
		--profile $(developer_profile)

.PHONY: walter-white-job-template-test
walter-white-job-template-test:
	aws cloudformation validate-template --region $(region) --profile $(developer_profile) --template-body file://./walter-white-job.yaml >> /dev/null

.PHONY: walter-white-job
walter-white-job:
	aws cloudformation deploy \
		--template-file walter-white-job.yaml \
		--stack-name "$(service_name)" \
		--region $(region) \
        --profile $(developer_profile)
		--parameter-overrides ServiceName=$(service_name) BucketUri="s3://$(bucket_name)" MLFlowStackName="$(project_name)-ehrmantraut-on-ec2"

.PHONY: update-walter-white-job
update-walter-white-job:
	aws cloudformation create-change-set \
		--change-set-name "$(service_name)-${now}" \
		--change-set-type UPDATE \
		--template-body file://./walter-white-job.yaml \
		--stack-name "$(service_name)" \
		--region $(region) \
		--capabilities CAPABILITY_NAMED_IAM \
		--profile $(developer_profile) \
		--parameters ParameterKey=ServiceName,ParameterValue=$(service_name) ParameterKey=BucketUri,ParameterValue="s3://$(bucket_name)" ParameterKey=MLFlowStackName,ParameterValue="$(project_name)-ehrmantraut-on-ec2"

	sleep $(wait_for_status_change)

	aws cloudformation execute-change-set \
		--change-set-name "$(service_name)-${now}" \
		--stack-name "$(service_name)" \
		--region $(region) \
		--profile $(developer_profile)

.PHONY: infrastructure
infrastructure: compute-environment walter-white-job

.PHONY: template-test
template-test: compute-environment-template-test walter-white-job-template-test
