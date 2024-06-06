output "azure-ai-services-id" {
  value = { for i, j in azurerm_cognitive_account.AllAzureAIServices : j.name => j.id }
}

output "azure-ai-services-endpoint" {
  value = { for i, j in azurerm_cognitive_account.AllAzureAIServices : j.name => j.endpoint }
}
