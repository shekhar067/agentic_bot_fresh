This is my power automate workflow. 1) Which call an api 2)return an msg, api response and an adaptive card with all response binding - now the ask is that everything all remains same will be same but they want to change the design of the adaptive card and api response also changed. I have attached the workflow solution json file.  and here is the old new schema details and adaptive design change old new(required)-----------------
 api response old schema
--------------------
{
    "type": "object",
    "properties": {
        "trackingDetailAndOrderStatusDto": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "salesOrderNumber": {},
                    "salesOrderItemNo": {},
                    "purchaseOrderNo": {},
                    "purchaseOrderDate": {},
                    "accountNumber": {},
                    "shipmentStatus": {},
                    "trackingInfo": {},
                    "materialDescription": {},
                    "materialNumber": {},
                    "orderQuantity": {},
                    "netValue": {},
                    "documentCurrency": {},
                    "eta": {},
                    "trackingId": {},
                    "createdDate": {}
                },
                "required": [
                    "salesOrderNumber",
                    "salesOrderItemNo",
                    "purchaseOrderNo",
                    "purchaseOrderDate",
                    "accountNumber",
                    "shipmentStatus",
                    "trackingInfo",
                    "materialDescription",
                    "materialNumber",
                    "orderQuantity",
                    "netValue",
                    "documentCurrency",
                    "eta",
                    "trackingId",
                    "createdDate"
                ]
            }
        },
        "message": {
            "type": "string"
        }
    }
}

----------------------
 api response new schema ( yet to change in powerautoamte flow)
---------------------------
{
    "type": "object",
    "properties": {
        "trackingDetailAndOrderStatusDto": {},
        "orderDetailListDto": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "soNumber": {
                        
                    },
                    "poNumber": {
                        
                    },
                    "soDate": {
                        
                    },
                    "trackingDetailAndOrderStatusDto": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "salesOrderNumber": {
                                   
                                },
                                "salesOrderItemNo": {
                                    
                                },
                                "purchaseOrderNo": {
                               
                                },
                                "purchaseOrderDate": {
                                
                                },
                                "accountNumber": {
                                
                                },
                                "shipmentStatus": {
                                  
                                },
                                "trackingInfo": {
                                
                                },
                                "materialDescription": {
                              
                                },
                                "materialNumber": {
                                    
                                },
                                "orderQuantity": {
                                    
                                },
                                "netValue": {
                                  
                                },
                                "documentCurrency": {
                                   
                                },
                                "eta": {
                                   
                                },
                                "trackingId": {
                                   
                                },
                                "createdDate": {
                                   
                                }
                            },
                            "required": [
                                "salesOrderNumber",
                                "salesOrderItemNo",
                                "purchaseOrderNo",
                                "purchaseOrderDate",
                                "accountNumber",
                                "shipmentStatus",
                                "trackingInfo",
                                "materialDescription",
                                "materialNumber",
                                "orderQuantity",
                                "netValue",
                                "documentCurrency",
                                "eta",
                                "trackingId",
                                "createdDate"
                            ]
                        }
                    }
                },
                "required": [
                    "soNumber",
                    "poNumber",
                    "soDate",
                    "trackingDetailAndOrderStatusDto"
                ]
            }
        },
        "message": {
            "type": "string"
        }
    }
}




Page 1 of  2(Outer Paginations Max 5 items in a page)


------------------------------
New Design required
-----------------------------------

Old Design
----------------------------------------------------

SO No 	Po No   Expand/Coallapse

123		456		+ or -(for expand collapse)
121212  5984985498   -(this is expanded)
		
		so date : 24/04/2025 ( common for all SO )
		order vlaue: 456 USD  ( common for all  SO)
		material no of 1st so item : tracking id + tracking url(SO Item Specific)
        material no of 2nd so item : tracking id + tracking url(SO Item Specific)

Page 1 of  2(Outer Paginations Max 5 items in a page)
----------------
I need the data of the innder pagination in blow format | SO No                       | PO No      | Expand/Collapse |
\| --------------------------- | ---------- | --------------- |
\| 6308480                     | 4735899... | + or -          |
\| (expanded view below)       |            |                 |
\| • SO Date: 26/08/2024       |            |                 |
\| • Order Value: 35541.70 DKK |            |                 |
\| • Items:                    |            |                 |

```
 - SO Date : date
 - SO amount: currency+ amount 
 -795233(Material No of 1st item ) : Tracking ID + URL
 -795232(Material No of 2nd item ) : Tracking ID + URL
```

@OR(
    equals(mod(outputs('Count_SOs_Processed_For_OuterPageCheck'), 5), 0), 
    equals(outputs('Count_SOs_Processed_For_OuterPageCheck'), outputs('Get_Total_SO_Count_Overall'))
)



---------------------------------------
this is is the below new api resposne format: -- -- -- -- -- - {
  {
    "trackingDetailAndOrderStatusDto": null,
    "orderDetailListDto": [
        {
            "soNumber": "6308571172",
            "poNumber": "14-1175287-KH",
            "soDate": "2024-10-14",
            "trackingDetailAndOrderStatusDto": [
                {
                    "salesOrderNumber": "6308571172",
                    "salesOrderItemNo": "000010",
                    "purchaseOrderNo": "14-1175287-KH",
                    "purchaseOrderDate": "2024-10-09",
                    "accountNumber": "0094028401",
                    "shipmentStatus": "Open",
                    "trackingInfo": "Not Available",
                    "materialDescription": "RP-Filter, Cooling Fan, Pkg/5, V60",
                    "materialNumber": "453561507301",
                    "orderQuantity": "2",
                    "netValue": "23.30",
                    "documentCurrency": "USD",
                    "eta": "Not Available",
                    "trackingId": "Not Available",
                    "createdDate": "2024-10-14"
                },
                {
                    "salesOrderNumber": "6308571172",
                    "salesOrderItemNo": "000020",
                    "purchaseOrderNo": "14-1175287-KH",
                    "purchaseOrderDate": "2024-10-09",
                    "accountNumber": "0094028401",
                    "shipmentStatus": "Open",
                    "trackingInfo": "Not Available",
                    "materialDescription": "RP-Filter, Air Inlet, Pkg/5,V60/V60 Plus",
                    "materialNumber": "453561505991",
                    "orderQuantity": "2",
                    "netValue": "96.08",
                    "documentCurrency": "USD",
                    "eta": "Not Available",
                    "trackingId": "Not Available",
                    "createdDate": "2024-10-14"
                }
            ]
        },
        {
            "soNumber": "6308571172",
            "poNumber": "14-1175287-KH",
            "soDate": "2024-10-14",
            "trackingDetailAndOrderStatusDto": [
                {
                    "salesOrderNumber": "6308571172",
                    "salesOrderItemNo": "000010",
                    "purchaseOrderNo": "14-1175287-KH",
                    "purchaseOrderDate": "2024-10-09",
                    "accountNumber": "0094028401",
                    "shipmentStatus": "Open",
                    "trackingInfo": "Not Available",
                    "materialDescription": "RP-Filter, Cooling Fan, Pkg/5, V60",
                    "materialNumber": "453561507301",
                    "orderQuantity": "2",
                    "netValue": "23.30",
                    "documentCurrency": "USD",
                    "eta": "Not Available",
                    "trackingId": "Not Available",
                    "createdDate": "2024-10-14"
                },
                {
                    "salesOrderNumber": "6308571172",
                    "salesOrderItemNo": "000020",
                    "purchaseOrderNo": "14-1175287-KH",
                    "purchaseOrderDate": "2024-10-09",
                    "accountNumber": "0094028401",
                    "shipmentStatus": "Open",
                    "trackingInfo": "Not Available",
                    "materialDescription": "RP-Filter, Air Inlet, Pkg/5,V60/V60 Plus",
                    "materialNumber": "453561505991",
                    "orderQuantity": "2",
                    "netValue": "96.08",
                    "documentCurrency": "USD",
                    "eta": "Not Available",
                    "trackingId": "Not Available",
                    "createdDate": "2024-10-14"
                }
            ]
        }
    ],
    "message": "Record Found"
}

