# DVD Tools

A collection of tools for working with DVDs.

## ISO Browser

A plugin for browsing DVDs.
Supports everything `vssource.IsoFile` supports.
Loading an ISO file allows you to dump any title or angle to a file.

For more information,
see the [JET guide on DVD Remuxing](https://jaded-encoding-thaumaturgy.github.io/JET-guide/dvd-remux/sources/dvd-remux/)

Spot an issue?
Please leave an issue
with "ISO Browser" in the title.

### Dependencies

- `vs-source` and its dependencies (`pip install vsjetpack`, [dependencies](https://github.com/Jaded-Encoding-Thaumaturgy/vs-source?tab=readme-ov-file#how-to-install))
- `ffmpeg` (must be a build that supports `dvdread`, see [FFmpeg's docs](https://ffmpeg.org/ffmpeg-formats.html#dvdvideo))
- Unencrypted DVDs (this plugin does not decrypt DVDs)

### Features

- Load ISO and IFO files
- Navigate through titles and angles
- View title metadata, including audio tracks and chapters
- Preview video node (overrides the default output node)
- Dump titles and angles to files, with options to split by chapters
- Dump all titles/angles to files at once (no chapter splitting)
- Copy button for an IsoFile code snippet to use in your script

### Planned

- Navigation to chapters (depends on vspreview)
- Preview audio nodes (depends on vspreview)
- Implement some way to open vspreview without having to pre-load a video node (depends on vspreview)
- Remux VOBSub when dumping (depends on FFmpeg support)
- Toggle to render VOBSub (depends on IsoFile support)
- Dropdown to select subtitle track (depends on IsoFile support)

### Limitations

Any limitations of `vssource.IsoFile` and `FFmpeg` apply.
If you would like a feature to be supported,
please help implement it
in the relevant source code!

## SAR Helper

(Planned)
A plugin to help you figure out the SAR of your video.

### Features

- ...

### Planned

- ...
