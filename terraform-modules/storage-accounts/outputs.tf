output "sa-name" {
  value = { for i, j in azurerm_storage_account.AllSa : j.name => j.name }
}

output "sa-id" {
  value = { for i, j in azurerm_storage_account.AllSa : j.name => j.id }
}

output "sa-accesskey" {
  value = { for i, j in azurerm_storage_account.AllSa : j.name => j.primary_access_key }
}