"use strict";

(() => {
  const cover = document.querySelector(".cover");
  const background = document.getElementById("cover-media");
  if (!cover || !background) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const rows = [
    { family: "environment", clips: [["hy-worldplay", "case_007"], ["lingbot-v2", "case_016"], ["evoke", "case_036"], ["lingbot-world", "case_061"]] },
    { family: "object", clips: [["lingbot-v2", "case_047"], ["evoke", "case_033"], ["hy-worldplay", "case_054"], ["lingbot-world", "case_064"]] },
    { family: "human", clips: [["evoke", "case_038"], ["hy-worldplay", "case_054"], ["lingbot-v2", "case_048"], ["lingbot-world", "case_057"]] },
  ];
  const videos = [];

  for (const [index, { family, clips }] of rows.entries()) {
    const row = document.createElement("div");
    row.className = `marquee-row marquee-row-${index + 1}`;
    const track = document.createElement("div");
    track.className = "marquee-track";

    for (let repeat = 0; repeat < 2; repeat++) {
      const group = document.createElement("div");
      group.className = "marquee-group";
      for (const [model, caseName] of clips) {
        const assetName = `${family}-${model}-${caseName}`;
        const poster = `assets/cover/${assetName}.jpg`;
        const media = document.createElement(reducedMotion ? "img" : "video");
        media.className = "marquee-media";
        if (reducedMotion) {
          media.src = poster;
          media.alt = "";
        } else {
          media.src = `videos/cover/${assetName}.mp4`;
          media.poster = poster;
          media.loop = true;
          media.muted = true;
          media.playsInline = true;
          media.preload = "none";
          videos.push(media);
        }
        group.append(media);
      }
      track.append(group);
    }
    row.append(track);
    background.append(row);
  }

  if (videos.length) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        const video = entry.target;
        if (entry.isIntersecting) video.play().catch(() => {});
        else video.pause();
      }
    }, { threshold: 0.05 });
    videos.forEach(video => observer.observe(video));
  }
})();
