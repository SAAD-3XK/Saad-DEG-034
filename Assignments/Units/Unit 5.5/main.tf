terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 3.20.0"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = var.region
}

module "s3_module" {
  source      = "./s3_module"
  bucket_name = var.bucket_name
}

output "s3_bucket_id" {
  value = module.s3_module.s3_bucket_name
}