import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const p = await PresentationFile.importPptx(await FileBlob.load("/Users/user/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-dark-mode/assets/reference.pptx"));
console.log('presentation keys', Object.getOwnPropertyNames(Object.getPrototypeOf(p)));
console.log('slides keys', Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides)));
console.log('slide keys', Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides.getItem(0))));
console.log('slides own', Object.keys(p.slides));
