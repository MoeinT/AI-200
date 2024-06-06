import-module servermanager
add-windowsfeature web-server -includeallsubfeature
set-content -path "C:\inetpub\wwwroot\Default.html" -Value "<h1>This is the server on Vnet 2</h1>"