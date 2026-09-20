import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "/Users/user/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-dark-mode/assets/reference.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot = await presentation.inspect({
  kind: "slide,textbox,shape,image,table,chart,layout",
  maxChars: 40000,
});
console.log(snapshot.ndjson);
