module "AzureAIServices" {
  source = "../../../../terraform-modules/azure-ai-services"
  properties = {
    "azure-ai-services-cv-ai-200-${var.env}" = {
      location            = module.Rg.rg-locations["ai-102-${var.env}"],
      resource_group_name = module.Rg.rg-names["ai-102-${var.env}"],
      sku_name            = "S0"
      kind                = "CognitiveServices"
    }
  }
}
