terraform {
  required_version = ">= 1.0"
  required_providers {
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}

resource "null_resource" "deploy_gpu_service" {
  provisioner "local-exec" {
    command = "gcloud run services replace aisetup.yaml --region=us-central1"
  }
}


