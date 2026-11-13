# Exporta la presentación a PDF con PowerPoint (para revisión visual).
param([string]$Pptx = "$PSScriptRoot\Sustentacion_Proyecto_Electiva.pptx")
$pdf = [System.IO.Path]::ChangeExtension($Pptx, ".pdf")
$app = New-Object -ComObject PowerPoint.Application
try {
  $deck = $app.Presentations.Open($Pptx, $true, $false, $false)
  $deck.SaveAs($pdf, 32)  # 32 = ppSaveAsPDF
  $deck.Close()
} finally {
  $app.Quit()
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($app) | Out-Null
}
Write-Output $pdf
