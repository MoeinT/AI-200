module "AzureAIServices" {
  source = "../../../../terraform-modules/azure-ai-services"
  properties = {
    "azure-ai-services-ai-200-${var.env}" = {
      location            = module.Rg.rg-locations["ai-102-${var.env}"],
      resource_group_name = module.Rg.rg-names["ai-102-${var.env}"],
      kind                = "CognitiveServices"
      sku_name            = "S0"
    }
  }
}
