"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { LatLng, LatLngBounds, Map } from "leaflet";
import { IoClose, IoMenu } from "react-icons/io5";
import { FaX } from "react-icons/fa6";
import { AnimatePresence, motion } from "framer-motion";
const OpenStreetMap = dynamic(() => import("@/components/OpenStreetMap"), {
  ssr: false,
});

export default function Home() {
  const [mapState, setMapState] = useState<Map | null>(null);
  const [bounds, setBounds] = useState<LatLngBounds | undefined>(() => {
    return mapState?.getBounds();
  });

  useEffect(() => {
    mapState?.on("move", () => {
      setBounds(mapState?.getBounds());
    });
  }, [mapState]);

  return (
    <div className="flex flex-col grow overflow-hidden h-full">
      <OpenStreetMap
        setRef={(ref) => {
          setMapState(ref);
        }}
      />
      <MenuButton mapState={mapState} />
    </div>
  );
}

function MenuButton({ mapState }: { mapState: Map | null }) {
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
      <Legend mapState={mapState} zoomFunction={zoomFunction} />
      <AnimatePresence>
        {showMenu && <Menu mapState={mapState} />}
      </AnimatePresence>
    </>
  );
}

function Legend({
  mapState,
  zoomFunction,
}: {
  mapState: Map | null;
  zoomFunction: () => void;
}) {
  return (
    <>
      <div className="absolute bottom-3 left-3 z-[9999] w-[300px]">
        <button
          className="px-3 py-1 rounded-lg bg-blue-400 shadow-md my-2"
          onClick={() => zoomFunction()}
        >
          Zoom
        </button>
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

function Menu({ mapState }: { mapState: Map | null }) {
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
        <IoClose size={25} />
      </div>
    </motion.div>
  );
}
