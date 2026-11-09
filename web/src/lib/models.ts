/** "juggernautXL_v9.safetensors" -> "juggernautXL_v9" */
export function modelDisplayName(name: string): string {
  return name.replace(/\.safetensors$/, '')
}
