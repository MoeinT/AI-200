terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 3.0.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_cognitive_account" "AllAzureAIServices" {
  for_each            = var.properties
  name                = can(each.value.name) ? each.value.name : each.key
  location            = each.value.location
  resource_group_name = each.value.resource_group_name
  kind                = each.value.kind
  sku_name            = each.value.sku_name
  tags                = contains(keys(each.value), "tags") ? merge(local.DefaultTags, each.value.tags) : local.DefaultTags
}
