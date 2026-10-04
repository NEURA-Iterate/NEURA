<script lang="ts">
  import type { Snippet } from 'svelte'

  let {
    onfiles,
    multiple = false,
    disabled = false,
    filled = false,
    compact = false,
    children,
  }: {
    onfiles: (files: File[]) => void
    multiple?: boolean
    disabled?: boolean
    filled?: boolean
    compact?: boolean
    children: Snippet
  } = $props()

  let over = $state(false)
  let depth = 0
  let input: HTMLInputElement

  const isTiff = (f: File) => /\.tiff?$/i.test(f.name)

  function take(list: FileList | null | undefined) {
    const files = Array.from(list ?? []).filter(isTiff)
    if (files.length) onfiles(multiple ? files : files.slice(0, 1))
  }
  function enter(e: DragEvent) {
    e.preventDefault()
    if (disabled) return
    depth++
    over = true
  }
  function leave() {
    depth = Math.max(0, depth - 1)
    if (!depth) over = false
  }
  function drop(e: DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    depth = 0
    over = false
    if (!disabled) take(e.dataTransfer?.files)
  }
</script>

<div
  class="drop"
  class:over
  class:filled
  class:compact
  class:disabled
  role="button"
  tabindex={disabled ? -1 : 0}
  aria-disabled={disabled}
  onclick={() => !disabled && input.click()}
  onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && (e.preventDefault(), input.click())}
  ondragenter={enter}
  ondragover={(e) => e.preventDefault()}
  ondragleave={leave}
  ondrop={drop}
>
  {@render children()}
  <input
    bind:this={input}
    type="file"
    accept=".tif,.tiff"
    {multiple}
    {disabled}
    hidden
    onchange={(e) => { take(e.currentTarget.files); e.currentTarget.value = '' }}
  />
</div>

<style>
  .drop {
    position: relative;
    display: grid;
    place-items: center;
    text-align: center;
    gap: 6px;
    padding: 28px 20px;
    border: 1.5px dashed #c9ced6;
    border-radius: 18px;
    background: #fafbfc;
    color: #555;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
  }
  .drop.compact { padding: 16px 14px; border-radius: 14px; place-items: stretch; text-align: left; }
  .drop:hover, .drop:focus-visible { border-color: #8fa9d6; background: #f5f8fd; outline: none; }
  .drop.over { border-color: #1f5fbf; border-style: solid; background: #eef4ff; box-shadow: 0 0 0 4px #1f5fbf1a; }
  .drop.filled { border-style: solid; border-color: #d7e6da; background: #f6fbf7; }
  .drop.disabled { opacity: 0.55; cursor: default; }
</style>
