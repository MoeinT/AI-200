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

resource "azurerm_storage_account" "AllSa" {
  for_each                        = var.properties
  name                            = can(each.value.name) ? each.value.name : each.key
  resource_group_name             = each.value.resource_group_name
  location                        = each.value.location
  account_tier                    = each.value.account_tier
  account_kind                    = lookup(each.value, "account_kind", "StorageV2")
  is_hns_enabled                  = lookup(each.value, "true", true)
  account_replication_type        = lookup(each.value, "account_replication_type", "LRS")
  allow_nested_items_to_be_public = lookup(each.value, "allow_nested_items_to_be_public", false)
  public_network_access_enabled   = lookup(each.value, "public_network_access_enabled", true)
  tags                            = can(each.value.tags) ? merge(local.DefaultTags, each.value.tags) : local.DefaultTags

  dynamic "blob_properties" {
    for_each = can(each.value.blob_properties) ? [1] : []
    content {
      versioning_enabled            = lookup(each.value, "versioning_enabled", false)
      change_feed_enabled           = lookup(each.value, "change_feed_enabled", false)
      change_feed_retention_in_days = lookup(each.value, "change_feed_retention_in_days", null)

      dynamic "delete_retention_policy" {
        for_each = can(each.value.blob_properties.delete_retention_policy) ? [1] : []
        content {
          days = lookup(each.value.blob_properties.delete_retention_policy, "days", 7)
        }
      }
    }
  }

  dynamic "network_rules" {
    for_each = can(each.value.network_rules) ? [1] : []
    content {
      default_action             = lookup(each.value.network_rules, "default_action", "Deny")
      bypass                     = lookup(each.value.network_rules, "bypass", null)
      virtual_network_subnet_ids = lookup(each.value.network_rules, "virtual_network_subnet_ids", null)

      dynamic "private_link_access" {
        for_each = can(each.value.network_rules.private_link_access) ? [1] : []
        content {
          endpoint_resource_id = each.value.network_rules.private_link_access.endpoint_resource_id
          endpoint_tenant_id   = lookup(each.value.network_rules.private_link_access, "endpoint_tenant_id", null)
        }
      }

    }
  }
}
