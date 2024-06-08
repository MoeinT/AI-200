module "Rg" {
  source = "../../../../terraform-modules/resource-groups"
  env    = var.env
  properties = {
    "ai-102-${var.env}" = {
      location = "West Europe"
      tags     = { "Terraform_Developer" : "Moein" }
    }
  }
}
