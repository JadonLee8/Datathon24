"use client";
import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { LatLngBounds, Map } from "leaflet";
import { IoClose, IoMenu } from "react-icons/io5";
import { AnimatePresence, motion } from "framer-motion";
import ToastManager from "@/components/toast/ToastManager";
import { ImageOverlayProps } from "react-leaflet";
import { fetchUtil } from "@/app/fetch";

const OpenStreetMap = dynamic(() => import("@/components/OpenStreetMap"), {
  ssr: false,
});

export default function Home() {
  const [mapState, setMapState] = useState<Map | null>(null);
  const [bounds, setBounds] = useState<LatLngBounds | undefined>(() => {
    return mapState?.getBounds();
  });
  const [imageBounds, setImageBounds] = useState<LatLngBounds | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  useEffect(() => {
    mapState?.on("move", () => {
      setBounds(mapState?.getBounds());
    });
  }, [mapState]);

  const processImage = async () => {
    ToastManager.addToast("Processing Image...", "success", 1000);

    const boundsCurr = mapState?.getBounds()!;
    const boundsSend = [
      boundsCurr.getSouthEast().lat,
      boundsCurr.getNorthWest().lng,
      boundsCurr.getNorthWest().lat,
      boundsCurr.getSouthEast().lng,
    ];

    const response = await fetchUtil("http://127.0.0.1:5000/process-image", {
      method: "POST",
      body: {
        bounds: boundsSend,
      },
    });
    const res = await response.json();
    console.log(res);

    setImageBounds(bounds!);
    setImageUrl(res.image_url);
  };

  return (
    <div className="flex flex-col grow overflow-hidden h-full">
      <OpenStreetMap
        setRef={(ref) => {
          setMapState(ref);
        }}
        imageOverlay={
          imageBounds &&
          ({
            bounds: [
              [imageBounds?.getNorth(), imageBounds?.getEast()],
              [imageBounds?.getSouth(), imageBounds?.getWest()],
            ],
            url: imageUrl || "",
          } as ImageOverlayProps)
        }
      />
      <MenuButton mapState={mapState} processImageFunction={processImage} />
    </div>
  );
}

function MenuButton({
  mapState,
  processImageFunction,
}: {
  mapState: Map | null;
  processImageFunction: () => void;
}) {
  const [showMenu, setShowMenu] = useState(false);

  const zoomFunction = () => {
    mapState?.setZoom(10);
  };
  useEffect(() => {
    mapState?.on("click", (e) => {
      setShowMenu(false);
    });
  }, [mapState]);
  return (
    <>
      <div className="absolute z-[9999] top-0 right-0">
        <button
          className="relative p-3 rounded-lg bg-white top-3 right-3 shadow-md"
          onClick={() => setShowMenu(true)}
        >
          <IoMenu size={20} />
        </button>
      </div>
      <Legend
        mapState={mapState}
        zoomFunction={zoomFunction}
        processImgFunction={processImageFunction}
      />
      <AnimatePresence>
        {showMenu && (
          <Menu mapState={mapState} closeMenu={() => setShowMenu(false)} />
        )}
      </AnimatePresence>
    </>
  );
}

function Legend({
  mapState,
  zoomFunction,
  processImgFunction,
}: {
  mapState: Map | null;
  zoomFunction: () => void;
  processImgFunction: () => void;
}) {
  const [zoom, setZoom] = useState<number>(0);

  useEffect(() => {
    mapState?.on("zoom", () => {
      console.log("Zoom: " + mapState?.getZoom());
      setZoom(mapState?.getZoom() || 0);
    });
  }, [mapState]);

  return (
    <>
      <div className="absolute bottom-3 left-3 z-[9999] w-[300px]">
        <div className="flex gap-2">
          <button
            className="px-3 py-1 rounded-lg bg-blue-400 shadow-md my-2"
            onClick={() => zoomFunction()}
          >
            Zoom
          </button>

          <button
            className={`px-3 py-1 rounded-lg ${zoom === 10 ? "bg-green-400" : "bg-gray-400"} shadow-md my-2`}
            onClick={() => {
              if (zoom === 10) {
                processImgFunction();
              } else {
                ToastManager.addToast(
                  "Must zoom to correct level first",
                  "error",
                  1000,
                );
              }
            }}
          >
            Process Image
          </button>
        </div>
        <div className="bg-gray-100 rounded-md p-3 ">
          <p className="text-black font-semibold text-sm text-wrap">
            Bounds: ({mapState?.getBounds().getNorth().toFixed(2)},
            {mapState?.getBounds().getEast().toFixed(2)}) to (
            {mapState?.getBounds().getSouth().toFixed(2)},
            {mapState?.getBounds().getWest().toFixed(2)})
          </p>
          <p className="text-black font-semibold text-sm text-wrap">
            Center: ({mapState?.getCenter().lat?.toFixed(2)},
            {mapState?.getCenter().lng.toFixed(2)})
          </p>
        </div>
      </div>
    </>
  );
}

function Menu({
  mapState,
  closeMenu,
}: {
  mapState: Map | null;
  closeMenu: () => void;
}) {
  return (
    <motion.div
      initial={{ x: 288, opacity: 0.9 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 288, opacity: 0.9 }}
      transition={{ duration: 0.2, type: "linear" }}
      className="absolute right-0 z-[9999] w-72 h-screen bg-gray-200 overflow-hidden shadow-md rounded-l-md"
    >
      <div className="flex p-2 items-center">
        <div className="font-bold text-md grow">Cotton Image Segmentation</div>
        <button onClick={closeMenu}>
          <IoClose size={25} />
        </button>
      </div>
    </motion.div>
  );
}
