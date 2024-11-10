"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { LatLng, LatLngBounds, Map } from "leaflet";
const OpenStreetMap = dynamic(() => import("@/components/OpenStreetMap"), {
  ssr: false,
});

export default function Home() {
  const [mapState, setMapState] = useState<Map | null>(null);
  const [bounds, setBounds] = useState<LatLng | undefined>(() =>
    mapState?.getCenter(),
  );

  useEffect(() => {
    console.log("MapState", mapState);
    mapState?.on("move", () => {
      setBounds(mapState?.getCenter());
    });
    mapState?.getBounds();
  }, [mapState]);

  return (
    <div className="grow bg-lightmaroon">
      <h1 className="text-2xl font-semibold">Hello</h1>
      <OpenStreetMap
        setRef={(ref) => {
          setMapState(ref);
        }}
      />
      <h3 className="text-white">{mapState?.getBounds().toBBoxString()}</h3>
      <button></button>
    </div>
  );
}
